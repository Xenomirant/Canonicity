"""Run dense segment analysis from an existing completed samples.jsonl."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .cli import parse_lengths
from .core import SampledSequence
from .segment_reporting import write_segment_report
from .segments import analyze_segments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact dense prefix segments from a completed canonicity "
            "samples.jsonl without sampling the model again."
        )
    )
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Defaults to metadata.json beside --samples-file",
    )
    parser.add_argument(
        "--recurrence-horizons",
        type=parse_lengths,
        default=parse_lengths("128,256,512,1024,2048"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to the directory containing --samples-file",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser


def _load_samples(path: Path) -> tuple[SampledSequence, ...]:
    samples = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            key = (value["context_id"], int(value["sample_index"]))
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate rollout {key}")
            seen.add(key)
            samples.append(
                SampledSequence(
                    context_id=key[0],
                    sample_index=key[1],
                    token_ids=tuple(int(token) for token in value["generated_token_ids"]),
                    termination=str(value["termination"]),
                )
            )
    if not samples:
        raise ValueError(f"{path} contains no rollouts")
    return tuple(samples)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    metadata_path = args.metadata_file or args.samples_file.with_name("metadata.json")
    output = args.output or args.samples_file.parent
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    tokenizer_id = metadata["tokenizer"]
    revision = metadata.get("tokenizer_commit") or metadata.get("requested_revision")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install the project dependencies first") from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)
    samples = _load_samples(args.samples_file)

    def show_progress(done: int, total: int) -> None:
        interval = max(1, total // 20)
        if done == total or done % interval == 0:
            print(f"Segment analysis: {done}/{total} rollouts", flush=True)

    analysis = analyze_segments(
        tokenizer,
        samples,
        args.recurrence_horizons,
        progress=show_progress,
        workers=args.workers,
    )
    write_segment_report(output, analysis, source_samples=args.samples_file)
    print(f"Wrote segment analysis to {output}")


if __name__ == "__main__":
    main()
