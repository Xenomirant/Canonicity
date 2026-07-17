"""Sampling continuations from Hugging Face causal language models."""

import hashlib
import json
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core import SampledSequence, extract_continuation
from .sample_store import SampleBatchStore


SAMPLING_IMPLEMENTATION = "generated-text-canonicity/sampling-v2"
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class PromptContext:
    context_id: str
    text: Optional[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationSettings:
    model_id: str
    tokenizer_id: Optional[str]
    revision: Optional[str]
    samples_per_context: int
    max_new_tokens: int
    batch_size: int
    seed: int
    device: str
    dtype: str
    prompt_mode: str = "raw"
    device_map: Optional[str] = None
    tokenizer_revision: Optional[str] = None


def load_prompt_file(path: Path) -> Tuple[PromptContext, ...]:
    """Read JSONL prompts, each a string or an object with ``id`` and ``text``."""

    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, str):
                prompts.append(
                    PromptContext(
                        context_id=f"prompt-{len(prompts) + 1:03d}", text=value
                    )
                )
            elif isinstance(value, dict):
                context_id = value.get("id")
                text = value.get("text")
                if not isinstance(context_id, str) or not isinstance(text, str):
                    raise ValueError(
                        f"{path}:{line_number}: prompt objects need string id and text"
                    )
                metadata = value.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"{path}:{line_number}: prompt metadata must be an object"
                    )
                prompts.append(
                    PromptContext(
                        context_id=context_id,
                        text=text,
                        metadata=metadata,
                    )
                )
            else:
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON string or object"
                )
    _validate_prompts(prompts)
    return tuple(prompts)


def prompts_from_arguments(prompt_texts: Optional[Sequence[str]]) -> Tuple[PromptContext, ...]:
    if not prompt_texts:
        return (PromptContext(context_id="unconditional", text=None),)
    prompts = tuple(
        PromptContext(context_id=f"prompt-{index:03d}", text=text)
        for index, text in enumerate(prompt_texts, start=1)
    )
    _validate_prompts(prompts)
    return prompts


def _validate_prompts(prompts: Sequence[PromptContext]) -> None:
    if not prompts:
        raise ValueError("at least one prompt is required")
    identifiers = [prompt.context_id for prompt in prompts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prompt ids must be unique")


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
        except RuntimeError:
            pass
        else:
            return "mps"
    return "cpu"


def _resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    """Resolve an explicit cast or preserve the checkpoint's native dtype."""

    del device  # Device placement and weight precision are separate conditions.
    if requested == "auto":
        return "auto"
    return getattr(torch, requested)


def _unconditional_input_id(model: Any, tokenizer: Any) -> int:
    candidates = (
        getattr(model.generation_config, "bos_token_id", None),
        getattr(model.config, "bos_token_id", None),
        tokenizer.bos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
        tokenizer.eos_token_id,
    )
    for candidate in candidates:
        if isinstance(candidate, (tuple, list)):
            candidate = candidate[0] if candidate else None
        if candidate is not None:
            return int(candidate)
    raise ValueError(
        "unconditional generation needs a BOS or EOS seed token; provide a prompt"
    )


def _model_input(
    prompt: PromptContext,
    tokenizer: Any,
    model: Any,
    torch: Any,
    prompt_mode: str,
) -> Any:
    if prompt.text is None:
        token_id = _unconditional_input_id(model, tokenizer)
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    if prompt_mode == "chat":
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            prompt.text,
            add_special_tokens=True,
            return_tensors="pt",
        )
    if encoded["input_ids"].shape[-1] == 0:
        token_id = _unconditional_input_id(model, tokenizer)
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
    return encoded


def _context_window(config: Any) -> Optional[int]:
    for name in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Return only tokens that the generation configuration treats as EOS."""

    eos_value = getattr(model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = getattr(model.config, "eos_token_id", None)
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    if isinstance(eos_value, (list, tuple)):
        return {int(value) for value in eos_value}
    if eos_value is None:
        return set()
    return {int(eos_value)}


def _batch_seed(settings: GenerationSettings, context_id: str, start: int) -> int:
    """Derive a stable independent seed for one transactional generation batch."""

    material = json.dumps(
        {
            "scheme": "sha256-batch-v1",
            "base_seed": settings.seed,
            "model_id": settings.model_id,
            "context_id": context_id,
            "batch_start": start,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _resolve_repo_commit(hf_api: Any, repo_id: str, revision: Optional[str]) -> str:
    """Resolve a model-repository revision once, before loading any files."""

    if revision is not None and _FULL_COMMIT.fullmatch(revision):
        return revision.lower()
    info = hf_api.model_info(repo_id, revision=revision)
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or not _FULL_COMMIT.fullmatch(commit):
        raise RuntimeError(f"could not resolve an immutable commit for {repo_id}")
    return commit.lower()


def _load_model_exact(
    auto_model_class: Any,
    model_id: str,
    model_config: Any,
    model_kwargs: Mapping[str, Any],
) -> Any:
    """Load the checkpoint's native architecture and fail on ignored weights."""

    model, loading_info = auto_model_class.from_pretrained(
        model_id,
        config=model_config,
        output_loading_info=True,
        **model_kwargs,
    )
    load_failures = {
        name: loading_info.get(name, [])
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(name)
    }
    if load_failures:
        raise RuntimeError(
            "refusing to sample from incompletely loaded model weights: "
            f"{load_failures}"
        )
    return model


def _hardware_signature(torch: Any, device: str) -> Dict[str, Any]:
    """Record hardware properties that can affect sampled floating-point logits."""

    signature: Dict[str, Any] = {
        "resolved_device": device,
        "machine": platform.machine(),
    }
    if device.startswith("cuda"):
        signature["cuda_version"] = torch.version.cuda
        signature["cudnn_version"] = torch.backends.cudnn.version()
        signature["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    elif device == "mps":
        signature["mac_ver"] = platform.mac_ver()[0]
    else:
        signature["processor"] = platform.processor()
    return signature


def generate_samples(
    settings: GenerationSettings,
    prompts: Sequence[PromptContext],
    *,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[Any, Tuple[SampledSequence, ...], Dict[str, Any]]:
    """Sample independently for every context and return only continuation ids."""

    if settings.samples_per_context < 1:
        raise ValueError("samples_per_context must be positive")
    if settings.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if settings.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if settings.prompt_mode not in {"raw", "chat"}:
        raise ValueError("prompt_mode must be raw or chat")
    if settings.device_map is not None and settings.device != "auto":
        raise ValueError(
            "device_map and an explicit device are mutually exclusive placement modes"
        )
    _validate_prompts(prompts)

    try:
        import torch
        import accelerate
        import tokenizers
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "generation requires the project dependencies; install with `pip install -e .`"
        ) from error

    device = _resolve_device(torch, settings.device)
    dtype = _resolve_dtype(torch, settings.dtype, device)
    tokenizer_id = settings.tokenizer_id or settings.model_id
    hf_api = HfApi()
    model_commit = _resolve_repo_commit(
        hf_api, settings.model_id, settings.revision
    )
    if tokenizer_id == settings.model_id and settings.tokenizer_revision is None:
        tokenizer_commit = model_commit
    else:
        tokenizer_commit = _resolve_repo_commit(
            hf_api, tokenizer_id, settings.tokenizer_revision
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_commit,
    )
    model_config = AutoConfig.from_pretrained(
        settings.model_id,
        revision=model_commit,
    )
    model_kwargs = {
        "revision": model_commit,
        "dtype": dtype,
    }
    if settings.device_map is not None:
        model_kwargs["device_map"] = settings.device_map
    model = _load_model_exact(
        AutoModelForCausalLM,
        settings.model_id,
        model_config,
        model_kwargs,
    )
    if settings.device_map is None:
        model.to(device)
    model.eval()

    if settings.device_map is None:
        input_device = device
    else:
        input_device = model.get_input_embeddings().weight.device

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("batched generation needs an existing PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    terminal_ids = _eos_token_ids(model, tokenizer)

    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    hardware_signature = _hardware_signature(torch, device)
    resolved_device_map = {
        str(name): str(value)
        for name, value in getattr(model, "hf_device_map", {}).items()
    }
    context_config = getattr(model.config, "text_config", model.config)
    attention_implementation = getattr(
        context_config, "_attn_implementation", None
    )
    context_window = _context_window(context_config)
    prepared_contexts = []
    context_inputs = []
    for prompt in prompts:
        model_input = _model_input(
            prompt,
            tokenizer,
            model,
            torch,
            settings.prompt_mode,
        )
        model_input = {
            name: value.to(input_device)
            for name, value in model_input.items()
            if hasattr(value, "to")
        }
        input_width = int(model_input["input_ids"].shape[-1])
        if (
            context_window is not None
            and input_width + settings.max_new_tokens > context_window
        ):
            raise ValueError(
                f"context {prompt.context_id!r} needs "
                f"{input_width + settings.max_new_tokens} tokens but the "
                f"model context window is {context_window}"
            )
        raw_prompt_tokens = (
            0
            if prompt.text is None
            else len(tokenizer.encode(prompt.text, add_special_tokens=False))
        )
        context_inputs.append(
            {
                "id": prompt.context_id,
                "raw_prompt_tokens": raw_prompt_tokens,
                "model_input_tokens": input_width,
            }
        )
        prepared_contexts.append((prompt, model_input, input_width))

    store = None
    if checkpoint_dir is not None:
        sampling_plan = {
            "sampling_implementation": SAMPLING_IMPLEMENTATION,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "tokenizers_version": tokenizers.__version__,
            "accelerate_version": accelerate.__version__,
            "model_id": settings.model_id,
            "model_class": type(model).__name__,
            "tokenizer_id": tokenizer_id,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "requested_model_revision": settings.revision,
            "requested_tokenizer_revision": settings.tokenizer_revision,
            "model_commit": model_commit,
            "tokenizer_commit": tokenizer_commit,
            "samples_per_context": settings.samples_per_context,
            "max_new_tokens": settings.max_new_tokens,
            "batch_size": settings.batch_size,
            "base_seed": settings.seed,
            "seed_scheme": "sha256-batch-v1",
            "requested_device": settings.device,
            "resolved_device": device,
            "hardware_signature": hardware_signature,
            "requested_device_map": settings.device_map,
            "resolved_device_map": resolved_device_map,
            "requested_dtype": settings.dtype,
            "resolved_parameter_dtypes": parameter_dtypes,
            "attention_implementation": attention_implementation,
            "eos_token_ids": sorted(terminal_ids),
            "unconditional_seed_is_evaluated": False,
            "prompt_tokens_are_evaluated": False,
            "other_sampled_special_tokens_are_evaluated": True,
            "prompt_mode": settings.prompt_mode,
            "sampling": {
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "prompts": [
                {
                    "id": prompt.context_id,
                    "text": prompt.text,
                    "metadata": dict(prompt.metadata),
                }
                for prompt in prompts
            ],
            "context_inputs": context_inputs,
        }
        store = SampleBatchStore(checkpoint_dir, sampling_plan)

    samples: List[SampledSequence] = []
    with torch.inference_mode():
        for context_position, (
            prompt,
            model_input,
            input_width,
        ) in enumerate(prepared_contexts):
            produced = 0
            while produced < settings.samples_per_context:
                current_batch = min(
                    settings.batch_size, settings.samples_per_context - produced
                )
                completed = (
                    store.load_batch(
                        context_position,
                        prompt.context_id,
                        produced,
                        current_batch,
                    )
                    if store is not None
                    else None
                )
                if completed is not None:
                    samples.extend(completed)
                    produced += current_batch
                    continue
                batch_seed = _batch_seed(settings, prompt.context_id, produced)
                torch.manual_seed(batch_seed)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(batch_seed)
                outputs = model.generate(
                    **model_input,
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    max_new_tokens=settings.max_new_tokens,
                    num_return_sequences=current_batch,
                    pad_token_id=tokenizer.pad_token_id,
                )
                batch_samples = []
                for offset, row in enumerate(outputs.tolist()):
                    token_ids, termination = extract_continuation(
                        row, input_width, terminal_ids
                    )
                    batch_samples.append(
                        SampledSequence(
                            context_id=prompt.context_id,
                            sample_index=produced + offset,
                            token_ids=token_ids,
                            termination=termination,
                        )
                    )
                if store is not None:
                    store.write_batch(
                        context_position,
                        prompt.context_id,
                        produced,
                        batch_samples,
                    )
                samples.extend(batch_samples)
                produced += current_batch

    metadata = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
        "accelerate_version": accelerate.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "requested_device": settings.device,
        "resolved_device": device,
        "hardware_signature": hardware_signature,
        "requested_device_map": settings.device_map,
        "resolved_device_map": resolved_device_map,
        "input_device": str(input_device),
        "requested_dtype": settings.dtype,
        "resolved_parameter_dtypes": parameter_dtypes,
        "attention_implementation": attention_implementation,
        "model_commit": model_commit,
        "tokenizer_commit": tokenizer_commit,
        "model_config_type": type(model.config).__name__,
        "model_class": type(model).__name__,
        "native_composite_config_loaded": hasattr(model.config, "text_config"),
        "model_context_window": context_window,
        "context_inputs": context_inputs,
        "unconditional_seed_is_evaluated": False,
        "prompt_tokens_are_evaluated": False,
        "eos_token_ids": sorted(terminal_ids),
        "other_sampled_special_tokens_are_evaluated": True,
        "seed_scheme": "sha256-batch-v1",
        "sampling_implementation": SAMPLING_IMPLEMENTATION,
        "transactional_batch_checkpoints": checkpoint_dir is not None,
    }
    return tokenizer, tuple(samples), metadata
