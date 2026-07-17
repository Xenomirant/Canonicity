"""CLI for reproducibly deriving long prompts from Hugging Face datasets."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .dataset_prompts import (
    materialize_document_prompts,
    materialize_prompts,
    write_prompts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize token-budgeted JSONL prompts from sequential rows or "
            "distinct sampled documents."
        )
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    parser.add_argument("--config", help="Dataset configuration name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--field", default="text")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--separator", default="\n")
    parser.add_argument(
        "--document-start-regex",
        help=(
            "Select one prompt per document, where matching rows begin a new "
            "document; otherwise pack sequential rows"
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "dataset prompt materialization requires the project dependencies"
        ) from error

    api = HfApi()
    dataset_commit = api.dataset_info(
        args.dataset,
        revision=args.dataset_revision,
    ).sha
    tokenizer_commit = api.model_info(
        args.tokenizer,
        revision=args.tokenizer_revision,
    ).sha
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=tokenizer_commit,
    )
    rows = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        revision=dataset_commit,
        streaming=True,
    )
    provenance = {
        "dataset": args.dataset,
        "dataset_config": args.config,
        "dataset_split": args.split,
        "requested_dataset_revision": args.dataset_revision,
        "dataset_commit": dataset_commit,
        "field": args.field,
        "tokenizer": args.tokenizer,
        "requested_tokenizer_revision": args.tokenizer_revision,
        "tokenizer_commit": tokenizer_commit,
    }
    if args.document_start_regex:
        if args.skip_rows:
            raise ValueError(
                "--skip-rows cannot be combined with --document-start-regex"
            )
        prompts = materialize_document_prompts(
            rows,
            tokenizer,
            field=args.field,
            count=args.count,
            prompt_tokens=args.prompt_tokens,
            separator=args.separator,
            document_start_pattern=args.document_start_regex,
            seed=args.selection_seed,
            provenance=provenance,
        )
    else:
        prompts = materialize_prompts(
            rows,
            tokenizer,
            field=args.field,
            count=args.count,
            prompt_tokens=args.prompt_tokens,
            separator=args.separator,
            skip_rows=args.skip_rows,
            provenance=provenance,
        )
    write_prompts(args.output, prompts)
    print(f"Wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
