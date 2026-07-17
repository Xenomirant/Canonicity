"""Command-line interface for the generated-text canonicity experiment."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .core import evaluate_samples
from .generation import (
    GenerationSettings,
    load_prompt_file,
    prompts_from_arguments,
    generate_samples,
)
from .reporting import write_report
from .segment_reporting import write_segment_report
from .segments import analyze_segments


PAPER_URL = "https://arxiv.org/abs/2408.08541"


def parse_lengths(value: str) -> Tuple[int, ...]:
    """Parse integers and inclusive ``start:end[:step]`` ranges."""

    lengths = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("empty item in --lengths")
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise argparse.ArgumentTypeError(f"invalid length range: {part}")
            try:
                numbers = [int(piece) for piece in pieces]
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid length range: {part}"
                ) from error
            start, end = numbers[:2]
            step = numbers[2] if len(numbers) == 3 else 1
            if start < 1 or end < start or step < 1:
                raise argparse.ArgumentTypeError(f"invalid length range: {part}")
            lengths.update(range(start, end + 1, step))
            lengths.add(end)
        else:
            try:
                length = int(part)
            except ValueError as error:
                raise argparse.ArgumentTypeError(f"invalid length: {part}") from error
            if length < 1:
                raise argparse.ArgumentTypeError("lengths must be positive")
            lengths.add(length)
    if not lengths:
        raise argparse.ArgumentTypeError("at least one length is required")
    return tuple(sorted(lengths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample model continuations and measure exact sequence-level "
            "tokenization canonicity."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument(
        "--tokenizer", help="Hugging Face tokenizer id (defaults to --model)"
    )
    parser.add_argument("--revision", help="Model revision or commit")
    parser.add_argument(
        "--tokenizer-revision",
        help=(
            "Tokenizer revision or commit; same-repository tokenizers default "
            "to the resolved model commit"
        ),
    )
    parser.add_argument(
        "--samples", type=int, default=256, help="Samples per context (default: 256)"
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("1:128"),
        help=(
            "Lengths/ranges such as 1:128, 128:2048:128, or 8,16,32 "
            "(default: 1:128)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, mps, or a torch device"
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        help="Let Transformers/Accelerate shard the model across devices",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("raw", "chat"),
        default="raw",
        help="Use prompts as raw prefixes or wrap them with the tokenizer chat template",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Conditioning text; repeat for multiple independently reported contexts",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        help="Use exactly the first N records from --prompts-file",
    )
    prompt_group.add_argument(
        "--prompts-file",
        type=Path,
        help="JSONL containing prompt strings or {\"id\", \"text\"} objects",
    )
    parser.add_argument(
        "--examples-per-length",
        type=int,
        default=3,
        help="Non-canonical examples retained per context and length",
    )
    parser.add_argument(
        "--segment-analysis",
        action="store_true",
        help=(
            "Evaluate every generated prefix, count maximal non-canonical runs, "
            "and run fixed-landmark recurrence association tests"
        ),
    )
    parser.add_argument(
        "--recurrence-horizons",
        type=parse_lengths,
        help="Even survival horizons for recurrence tests",
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=1,
        help="CPU threads for exact dense prefix analysis (default: 1)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.device_map is not None and args.device != "auto":
        parser.error("--device-map and an explicit --device are mutually exclusive")
    if args.tokenizer_revision is not None and args.tokenizer is None:
        parser.error("--tokenizer-revision requires --tokenizer")
    prompts = (
        load_prompt_file(args.prompts_file)
        if args.prompts_file
        else prompts_from_arguments(args.prompts)
    )
    if args.prompt_limit is not None:
        if args.prompts_file is None:
            parser.error("--prompt-limit requires --prompts-file")
        if args.prompt_limit < 1:
            parser.error("--prompt-limit must be positive")
        if len(prompts) < args.prompt_limit:
            parser.error(
                f"--prompts-file has {len(prompts)} records, fewer than "
                f"--prompt-limit {args.prompt_limit}"
            )
        prompts = prompts[: args.prompt_limit]

    recurrence_horizons = args.recurrence_horizons
    if recurrence_horizons is not None and not args.segment_analysis:
        parser.error("--recurrence-horizons requires --segment-analysis")
    if args.segment_analysis and recurrence_horizons is None:
        recurrence_horizons = tuple(
            horizon
            for horizon in (32, 64, 128, 256, 512, 1024, 2048)
            if horizon <= max(args.lengths)
        )
        if not recurrence_horizons:
            maximum = max(args.lengths)
            recurrence_horizons = (maximum if maximum % 2 == 0 else maximum - 1,)
    if recurrence_horizons is not None and (
        min(recurrence_horizons) < 2
        or any(horizon % 2 for horizon in recurrence_horizons)
        or max(recurrence_horizons) > max(args.lengths)
    ):
        parser.error(
            "--recurrence-horizons must be even, at least 2, and no larger "
            "than the maximum requested length"
        )
    if args.segment_workers < 1:
        parser.error("--segment-workers must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    settings = GenerationSettings(
        model_id=args.model,
        tokenizer_id=args.tokenizer,
        revision=args.revision,
        samples_per_context=args.samples,
        max_new_tokens=max(args.lengths),
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        prompt_mode=args.prompt_mode,
        device_map=args.device_map,
        tokenizer_revision=args.tokenizer_revision,
    )

    tokenizer, samples, runtime_metadata = generate_samples(
        settings,
        prompts,
        checkpoint_dir=args.output,
    )

    print(
        "Starting canonicity evaluation: "
        f"{len(samples)} rollouts x {len(args.lengths)} requested lengths",
        flush=True,
    )

    def show_evaluation_progress(done: int, total: int) -> None:
        interval = max(1, (total + 19) // 20)
        if done == total or done % interval == 0:
            print(
                f"Canonicity evaluation: {done}/{total} rollouts complete; "
                f"{total - done} remaining",
                flush=True,
            )

    evaluation = evaluate_samples(
        tokenizer,
        samples,
        args.lengths,
        examples_per_length=args.examples_per_length,
        progress=show_evaluation_progress,
    )
    metadata = {
        "paper": PAPER_URL,
        "paper_figure": 5,
        "metric": "exact equality: sampled_ids == encode(decode(sampled_ids))",
        "canonical_reencoding_add_special_tokens": False,
        "decode_skip_special_tokens": False,
        "decode_clean_up_tokenization_spaces": False,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "requested_revision": args.revision,
        "requested_tokenizer_revision": args.tokenizer_revision,
        "samples_per_context": args.samples,
        "lengths": list(args.lengths),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "segment_analysis": args.segment_analysis,
        "recurrence_horizons": (
            list(recurrence_horizons) if recurrence_horizons is not None else []
        ),
        "segment_workers": args.segment_workers,
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
        **runtime_metadata,
    }
    print("Canonicity evaluation complete; writing report...", flush=True)
    write_report(args.output, evaluation, metadata)
    if args.segment_analysis:
        assert recurrence_horizons is not None

        print(
            "Starting dense segment analysis: "
            f"{len(samples)} rollouts; workers={args.segment_workers}",
            flush=True,
        )

        def show_progress(done: int, total: int) -> None:
            interval = max(1, (total + 19) // 20)
            if done == total or done % interval == 0:
                print(
                    f"Segment analysis: {done}/{total} rollouts complete; "
                    f"{total - done} remaining",
                    flush=True,
                )

        segment_analysis = analyze_segments(
            tokenizer,
            samples,
            recurrence_horizons,
            progress=show_progress,
            workers=args.segment_workers,
        )
        print("Segment analysis complete; writing report...", flush=True)
        write_segment_report(
            args.output,
            segment_analysis,
            source_samples=args.output / "samples.jsonl",
        )
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
