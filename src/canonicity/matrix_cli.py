"""Isolated one-model jobs for the planned canonicity experiment matrix."""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from .cli import parse_lengths
from .generation import load_prompt_file


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    default_dtype: str
    default_batch_size: int
    default_attention_implementation: str
    default_device_map: Optional[str] = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gemma3-4b-it": ModelSpec(
        alias="gemma3-4b-it",
        model_id="google/gemma-3-4b-it",
        default_dtype="auto",
        default_batch_size=1,
        default_attention_implementation="flash_attention_2",
    ),
    "qwen3-30b-a3b-instruct-2507": ModelSpec(
        alias="qwen3-30b-a3b-instruct-2507",
        model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        default_dtype="auto",
        default_batch_size=1,
        default_attention_implementation="flash_attention_2",
        default_device_map="auto",
    ),
    "gemma-2b-it": ModelSpec(
        alias="gemma-2b-it",
        model_id="google/gemma-2b-it",
        default_dtype="auto",
        default_batch_size=1,
        default_attention_implementation="flash_attention_2",
    ),
    "llama2-7b": ModelSpec(
        alias="llama2-7b",
        model_id="meta-llama/Llama-2-7b-hf",
        default_dtype="auto",
        default_batch_size=1,
        default_attention_implementation="flash_attention_2",
    ),
    "mamba-130m": ModelSpec(
        alias="mamba-130m",
        model_id="state-spaces/mamba-130m-hf",
        default_dtype="auto",
        default_batch_size=32,
        default_attention_implementation="not_applicable",
    ),
}

MATRIX_CONDITIONS = ("unconditional", "wikitext")
MATRIX_CHECKPOINTS = (32, 64, 128, 256, 512, 1024, 2048)
MATRIX_RECURRENCE_HORIZONS = (128, 256, 512, 1024, 2048)

WORKSPACE_PROMPTS_FILE = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "wikitext-2-train-articles-mamba-1024-100.jsonl"
)


def _effective_attention_implementation(
    args: argparse.Namespace,
    spec: ModelSpec,
) -> str:
    if spec.default_attention_implementation == "not_applicable":
        return spec.default_attention_implementation
    return args.attention_implementation or spec.default_attention_implementation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated exact-weight canonicity jobs for the five-model matrix. "
            "Invoke one model/condition per scheduler job for large experiments."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        help="Model alias; repeat to run several sequentially",
    )
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        choices=MATRIX_CONDITIONS,
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Required outside this project's workspace checkout",
    )
    parser.add_argument("--prompt-count", type=int, default=100)
    parser.add_argument("--unconditional-rollouts", type=int, default=32)
    parser.add_argument("--prompt-rollouts", type=int, default=64)
    parser.add_argument(
        "--checkpoints",
        type=parse_lengths,
        default=MATRIX_CHECKPOINTS,
    )
    parser.add_argument(
        "--recurrence-horizons",
        type=parse_lengths,
        default=MATRIX_RECURRENCE_HORIZONS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-revision",
        action="append",
        default=[],
        metavar="ALIAS=COMMIT",
        help="Pin one model/tokenizer revision; repeat for multiple aliases",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--segment-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
    )
    parser.add_argument(
        "--no-device-map",
        action="store_true",
        help="Disable a model's default device map (Qwen uses auto by default)",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16")
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("flash_attention_2", "sdpa"),
        help="Override the backend for selected attention-based models",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/model-matrix")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _command(
    args: argparse.Namespace, spec: ModelSpec, condition: str
) -> tuple[list[str], Path]:
    output = args.output_root / spec.alias / condition
    batch_size = args.batch_size or spec.default_batch_size
    dtype = args.dtype or spec.default_dtype
    device_map = (
        None
        if args.no_device_map
        else (args.device_map or spec.default_device_map)
    )
    samples = (
        args.unconditional_rollouts
        if condition == "unconditional"
        else args.prompt_rollouts
    )
    attention_implementation = _effective_attention_implementation(args, spec)
    command = [
        sys.executable,
        "-m",
        "canonicity.cli",
        "--model",
        spec.model_id,
        "--samples",
        str(samples),
        "--lengths",
        ",".join(str(value) for value in args.checkpoints),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--dtype",
        dtype,
        "--attention-implementation",
        attention_implementation,
        "--segment-analysis",
        "--segment-workers",
        str(args.segment_workers),
        "--recurrence-horizons",
        ",".join(str(value) for value in args.recurrence_horizons),
        "--examples-per-length",
        "1",
        "--output",
        str(output),
    ]
    if device_map is not None:
        command.extend(["--device-map", device_map])
    revision = args.model_revisions.get(spec.alias)
    if revision is not None:
        command.extend(["--revision", revision])
    if condition == "wikitext":
        command.extend(
            [
                "--prompts-file",
                str(args.prompts_file.resolve()),
                "--prompt-limit",
                str(args.prompt_count),
                "--prompt-mode",
                "raw",
            ]
        )
    return command, output


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    args.model_revisions = {}
    for value in args.model_revision:
        alias, separator, revision = value.partition("=")
        if (
            not separator
            or alias not in MODEL_SPECS
            or not revision
            or alias in args.model_revisions
        ):
            raise SystemExit(
                "--model-revision must be a unique known ALIAS=COMMIT pair"
            )
        args.model_revisions[alias] = revision
    if args.prompt_count < 1:
        raise SystemExit("--prompt-count must be positive")
    if min(args.unconditional_rollouts, args.prompt_rollouts) < 1:
        raise SystemExit("rollout counts must be positive")
    if args.segment_workers < 1:
        raise SystemExit("--segment-workers must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.device_map is not None and args.no_device_map:
        raise SystemExit("--device-map and --no-device-map are mutually exclusive")
    if max(args.recurrence_horizons) > max(args.checkpoints):
        raise SystemExit("recurrence horizons cannot exceed the last checkpoint")
    if any(horizon < 2 or horizon % 2 for horizon in args.recurrence_horizons):
        raise SystemExit("recurrence horizons must be even integers >= 2")
    if "wikitext" in args.condition:
        if args.prompts_file is None:
            if not WORKSPACE_PROMPTS_FILE.exists():
                raise SystemExit(
                    "--prompts-file is required because the workspace prompt "
                    "asset is not installed with this package"
                )
            args.prompts_file = WORKSPACE_PROMPTS_FILE
        prompts = load_prompt_file(args.prompts_file)
        if len(prompts) < args.prompt_count:
            raise SystemExit(
                f"{args.prompts_file} has {len(prompts)} prompts; "
                f"{args.prompt_count} requested"
            )

    aliases = tuple(MODEL_SPECS) if args.all_models else tuple(args.model)
    if args.attention_implementation is not None and all(
        MODEL_SPECS[alias].default_attention_implementation == "not_applicable"
        for alias in aliases
    ):
        raise SystemExit(
            "--attention-implementation cannot override models without attention"
        )
    unused_revisions = set(args.model_revisions) - set(aliases)
    if unused_revisions:
        raise SystemExit(
            "--model-revision supplied for unselected models: "
            + ", ".join(sorted(unused_revisions))
        )
    conditions = tuple(dict.fromkeys(args.condition))
    for alias in aliases:
        spec = MODEL_SPECS[alias]
        effective_device_map = (
            None
            if args.no_device_map
            else (args.device_map or spec.default_device_map)
        )
        if effective_device_map is not None and args.device != "auto":
            raise SystemExit(
                "an explicit --device cannot be combined with an active device "
                "map; use --no-device-map to place the whole model on one device"
            )

    jobs = tuple(
        (MODEL_SPECS[alias], condition)
        for alias in aliases
        for condition in conditions
    )
    for job_position, (spec, condition) in enumerate(jobs, start=1):
        command, output = _command(args, spec, condition)
        attention_implementation = _effective_attention_implementation(args, spec)
        if (
            output.exists()
            and any(output.iterdir())
            and not (output / "sampling_plan.json").exists()
        ):
            raise SystemExit(
                "refusing a non-empty output directory without a matching "
                f"durable sampling plan: {output}"
            )
        context_count = 1 if condition == "unconditional" else args.prompt_count
        rollouts_per_context = (
            args.unconditional_rollouts
            if condition == "unconditional"
            else args.prompt_rollouts
        )
        print(
            f"Matrix job {job_position}/{len(jobs)}: model={spec.alias}; "
            f"condition={condition}; contexts/prompts={context_count}; "
            f"attention_implementation={attention_implementation}; "
            f"rollouts_per_context={rollouts_per_context}; "
            f"total_rollouts={context_count * rollouts_per_context}; "
            f"jobs_after_this={len(jobs) - job_position}",
            flush=True,
        )
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
