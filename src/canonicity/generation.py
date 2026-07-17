"""Sampling continuations from Hugging Face causal language models."""

import hashlib
import json
import platform
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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


def _sampling_progress_message(
    *,
    context_position: int,
    context_count: int,
    context_id: str,
    context_completed: int,
    samples_per_context: int,
    total_completed: int,
    total_samples: int,
    unfinished_contexts: int,
    sampled_this_invocation: int,
) -> str:
    """Format one append-only progress line for scheduler logs."""

    remaining_samples = total_samples - total_completed
    percentage = 100.0 * total_completed / total_samples
    return (
        "Sampling progress: "
        f"context {context_position + 1}/{context_count} {context_id!r}; "
        f"{context_completed}/{samples_per_context} completed in context; "
        f"{total_completed}/{total_samples} completed overall ({percentage:.1f}%); "
        f"{remaining_samples} rollouts still require model sampling; "
        f"{unfinished_contexts} unfinished contexts/prompts; "
        f"{sampled_this_invocation} sampled this invocation"
    )


def _parameter_footprint_by_device(model: Any) -> Dict[str, Dict[str, int]]:
    """Summarize logical and resident parameter bytes by reported device."""

    footprint: Dict[str, Dict[str, int]] = {}
    for parameter in model.parameters():
        device = str(parameter.device)
        row = footprint.setdefault(
            device,
            {
                "tensors": 0,
                "parameters": 0,
                "logical_bytes": 0,
                "resident_bytes": 0,
            },
        )
        parameter_count = int(parameter.numel())
        logical_bytes = parameter_count * int(parameter.element_size())
        row["tensors"] += 1
        row["parameters"] += parameter_count
        row["logical_bytes"] += logical_bytes
        if device.split(":", 1)[0] != "meta":
            row["resident_bytes"] += logical_bytes
    return dict(sorted(footprint.items()))


def _placement_summary(
    parameter_footprint: Mapping[str, Any],
    resolved_device_map: Mapping[str, Any],
) -> str:
    """Describe actual accelerator, CPU, and offload placement plainly."""

    locations = {str(location).lower() for location in parameter_footprint}
    locations.update(str(location).lower() for location in resolved_device_map.values())

    uses_cuda = any(
        location == "cuda"
        or location.startswith("cuda:")
        or location.isdigit()
        for location in locations
    )
    uses_mps = any(
        location == "mps" or location.startswith("mps:")
        for location in locations
    )
    uses_cpu = "cpu" in locations
    uses_disk = "disk" in locations
    uses_meta = "meta" in locations

    components = []
    if uses_cuda:
        components.append("GPU (CUDA)")
    if uses_mps:
        components.append("GPU (MPS)")
    if uses_cpu:
        components.append("CPU")
    if uses_disk:
        components.append("disk offload")
    if uses_meta and not uses_disk:
        components.append("offloaded/meta tensors")

    recognized = {
        location
        for location in locations
        if location in {"cpu", "mps", "disk", "meta", "cuda"}
        or location.startswith(("cuda:", "mps:"))
        or location.isdigit()
    }
    components.extend(sorted(locations - recognized))
    return " + ".join(components) if components else "unknown"


def _scan_checkpointed_batches(
    store: SampleBatchStore,
    prompts: Sequence[PromptContext],
    samples_per_context: int,
    batch_size: int,
    *,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[
    Dict[Tuple[int, int], Tuple[SampledSequence, ...]],
    Tuple[int, ...],
]:
    """Validate expected batch slots and count work already durably complete."""

    batch_slots_per_context = (
        samples_per_context + batch_size - 1
    ) // batch_size
    total_slots = len(prompts) * batch_slots_per_context
    completed_batches = {}
    completed_by_context = [0 for _ in prompts]
    completed_rollouts = 0
    scanned_slots = 0

    for context_position, prompt in enumerate(prompts):
        for start in range(0, samples_per_context, batch_size):
            expected_count = min(batch_size, samples_per_context - start)
            completed = store.load_batch(
                context_position,
                prompt.context_id,
                start,
                expected_count,
            )
            if completed is not None:
                completed_batches[(context_position, start)] = completed
                completed_by_context[context_position] += expected_count
                completed_rollouts += expected_count
            scanned_slots += 1
            if progress is not None:
                progress(
                    scanned_slots,
                    total_slots,
                    completed_rollouts,
                )

    return completed_batches, tuple(completed_by_context)


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
    total_samples = len(prompts) * settings.samples_per_context
    print(
        "Sampling plan: "
        f"{len(prompts)} contexts/prompts x "
        f"{settings.samples_per_context} rollouts = {total_samples} total; "
        f"max_new_tokens={settings.max_new_tokens}; "
        f"batch_size={settings.batch_size}",
        flush=True,
    )
    print(
        f"Resolving immutable revisions for model {settings.model_id!r} "
        f"and tokenizer {tokenizer_id!r}...",
        flush=True,
    )
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
    print(
        f"Resolved revisions: model={model_commit}; "
        f"tokenizer={tokenizer_commit}",
        flush=True,
    )
    print(
        f"Loading model with requested device={settings.device!r}, "
        f"device_map={settings.device_map!r}, dtype={settings.dtype!r}...",
        flush=True,
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
    resolved_device_map = {
        str(name): str(value)
        for name, value in getattr(model, "hf_device_map", {}).items()
    }
    parameter_footprint = _parameter_footprint_by_device(model)
    mapped_module_counts = dict(
        sorted(Counter(resolved_device_map.values()).items())
    )
    placement_summary = _placement_summary(parameter_footprint, resolved_device_map)
    hardware_signature = _hardware_signature(torch, device)
    print(
        "Sampling placement: "
        f"actual_model_placement={placement_summary}; "
        f"model_class={type(model).__name__}; "
        f"requested_device={settings.device}; resolved_target={device}; "
        f"input_device={input_device}; "
        f"parameter_footprint_by_device={json.dumps(parameter_footprint)}; "
        f"device_map_modules={mapped_module_counts or 'none'}; "
        f"parameter_dtypes={parameter_dtypes}",
        flush=True,
    )
    context_config = getattr(model.config, "text_config", model.config)
    attention_implementation = getattr(
        context_config, "_attn_implementation", None
    )
    context_window = _context_window(context_config)
    prepared_contexts = []
    context_inputs = []
    print(
        f"Preflighting {len(prompts)} context(s) against the model context window...",
        flush=True,
    )
    preflight_interval = max(1, (len(prompts) + 9) // 10)
    for prompt_position, prompt in enumerate(prompts, start=1):
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
        if (
            prompt_position == len(prompts)
            or prompt_position % preflight_interval == 0
        ):
            print(
                f"Context preflight: {prompt_position}/{len(prompts)} complete; "
                f"{len(prompts) - prompt_position} remaining",
                flush=True,
            )

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
    checkpointed_batches: Dict[
        Tuple[int, int], Tuple[SampledSequence, ...]
    ] = {}
    completed_by_context = [0 for _ in prompts]
    if store is not None:
        total_batch_slots = len(prompts) * (
            (settings.samples_per_context + settings.batch_size - 1)
            // settings.batch_size
        )
        print(
            f"Scanning {total_batch_slots} expected checkpoint batch slot(s)...",
            flush=True,
        )
        scan_interval = max(1, (total_batch_slots + 9) // 10)

        def show_checkpoint_scan(
            scanned: int,
            total: int,
            completed_rollouts: int,
        ) -> None:
            if scanned == total or scanned % scan_interval == 0:
                print(
                    f"Checkpoint scan: {scanned}/{total} batch slots checked; "
                    f"{completed_rollouts}/{total_samples} rollouts already "
                    "checkpointed",
                    flush=True,
                )

        checkpointed_batches, completed_counts = _scan_checkpointed_batches(
            store,
            prompts,
            settings.samples_per_context,
            settings.batch_size,
            progress=show_checkpoint_scan,
        )
        completed_by_context = list(completed_counts)

    restored_rollouts = sum(completed_by_context)
    total_completed = restored_rollouts
    sampled_this_invocation = 0
    unfinished_contexts = sum(
        completed < settings.samples_per_context
        for completed in completed_by_context
    )
    print(
        "Sampling resume state: "
        f"{restored_rollouts}/{total_samples} rollouts already checkpointed; "
        f"{total_samples - total_completed} still require model sampling; "
        f"{unfinished_contexts} unfinished contexts/prompts",
        flush=True,
    )
    last_progress_at = time.monotonic()
    local_progress_interval = max(
        1, (settings.samples_per_context + 9) // 10
    )
    with torch.inference_mode():
        for context_position, (
            prompt,
            model_input,
            input_width,
        ) in enumerate(prepared_contexts):
            context_completed = completed_by_context[context_position]
            if context_completed == settings.samples_per_context:
                print(
                    f"Sampling context {context_position + 1}/{len(prompts)} "
                    f"{prompt.context_id!r}: all "
                    f"{settings.samples_per_context} rollouts restored; "
                    "no model sampling needed",
                    flush=True,
                )
            else:
                print(
                    f"Sampling context {context_position + 1}/{len(prompts)} "
                    f"{prompt.context_id!r}: {context_completed}/"
                    f"{settings.samples_per_context} already checkpointed; "
                    f"{settings.samples_per_context - context_completed} require "
                    "model sampling",
                    flush=True,
                )
            produced = 0
            last_local_progress_bucket = (
                context_completed // local_progress_interval
            )
            while produced < settings.samples_per_context:
                current_batch = min(
                    settings.batch_size, settings.samples_per_context - produced
                )
                completed = checkpointed_batches.pop(
                    (context_position, produced),
                    None,
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

                total_completed += current_batch
                sampled_this_invocation += current_batch
                completed_by_context[context_position] += current_batch
                context_completed = completed_by_context[context_position]
                unfinished_contexts = sum(
                    completed_count < settings.samples_per_context
                    for completed_count in completed_by_context
                )
                now = time.monotonic()
                local_progress_bucket = (
                    context_completed // local_progress_interval
                )
                should_report = (
                    context_completed == settings.samples_per_context
                    or local_progress_bucket > last_local_progress_bucket
                    or now - last_progress_at >= 30.0
                )
                if should_report:
                    print(
                        _sampling_progress_message(
                            context_position=context_position,
                            context_count=len(prompts),
                            context_id=prompt.context_id,
                            context_completed=context_completed,
                            samples_per_context=settings.samples_per_context,
                            total_completed=total_completed,
                            total_samples=total_samples,
                            unfinished_contexts=unfinished_contexts,
                            sampled_this_invocation=sampled_this_invocation,
                        ),
                        flush=True,
                    )
                    last_local_progress_bucket = local_progress_bucket
                    last_progress_at = now

    if checkpointed_batches:
        raise RuntimeError("checkpoint scan left unconsumed sample batches")
    print(
        "Sampling complete: "
        f"{total_completed}/{total_samples} rollouts available; "
        f"{restored_rollouts} restored from checkpoints; "
        f"{sampled_this_invocation} sampled this invocation",
        flush=True,
    )

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
        "actual_model_placement": placement_summary,
        "parameter_footprint_by_device": parameter_footprint,
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
