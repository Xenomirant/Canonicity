"""Stable, inspectable experiment outputs."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .core import CANONICITY_EVALUATION_IMPLEMENTATION, EvaluationResult
from .sample_store import plan_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    output_dir: Path,
    evaluation: EvaluationResult,
    metadata: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(output_dir / "metadata.json", metadata)
    _write_summary(output_dir / "summary.csv", evaluation.summaries)
    if len({row.context_id for row in evaluation.summaries}) > 1:
        _write_summary(
            output_dir / "pooled_summary.csv",
            evaluation.pooled_summaries,
        )
    _write_sequences(output_dir / "samples.jsonl", evaluation)
    _write_examples(output_dir / "noncanonical_examples.jsonl", evaluation)
    _write_plot(output_dir / "canonicity.png", evaluation)
    sampling_plan_fingerprint = None
    sampling_plan_path = output_dir / "sampling_plan.json"
    if sampling_plan_path.exists():
        with sampling_plan_path.open("r", encoding="utf-8") as handle:
            sampling_manifest = json.load(handle)
        plan = sampling_manifest.get("plan")
        if (
            sampling_manifest.get("schema_version") != 1
            or not isinstance(plan, dict)
            or sampling_manifest.get("plan_fingerprint")
            != plan_fingerprint(plan)
        ):
            raise ValueError(f"invalid sampling plan: {sampling_plan_path}")
        sampling_plan_fingerprint = sampling_manifest["plan_fingerprint"]
    artifact_names = [
        "metadata.json",
        "summary.csv",
        "samples.jsonl",
        "noncanonical_examples.jsonl",
    ]
    if (output_dir / "pooled_summary.csv").exists():
        artifact_names.append("pooled_summary.csv")
    manifest = {
        "schema_version": 1,
        "evaluation_implementation": CANONICITY_EVALUATION_IMPLEMENTATION,
        "sampling_plan_fingerprint": sampling_plan_fingerprint,
        "rollouts": len(evaluation.sequences),
        "artifacts": {
            name: {"sha256": _sha256(output_dir / name)}
            for name in artifact_names
        },
    }
    _write_metadata(output_dir / "evaluation_manifest.json", manifest)


def _write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_summary(path: Path, summaries: Any) -> None:
    fieldnames = [
        "context_id",
        "length",
        "total_samples",
        "eligible_sequences",
        "terminated_before_length",
        "canonical_sequences",
        "noncanonical_sequences",
        "canonical_percentage",
        "canonical_ci95_low",
        "canonical_ci95_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            for field in (
                "canonical_percentage",
                "canonical_ci95_low",
                "canonical_ci95_high",
            ):
                if row[field] is None:
                    row[field] = ""
            writer.writerow(row)


def _write_sequences(path: Path, evaluation: EvaluationResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sequence in evaluation.sequences:
            record = {
                "context_id": sequence.sample.context_id,
                "sample_index": sequence.sample.sample_index,
                "generated_token_ids": list(sequence.sample.token_ids),
                "termination": sequence.sample.termination,
                "prefix_canonicity": {
                    str(prefix.length): prefix.is_canonical
                    for prefix in sequence.prefixes
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_examples(path: Path, evaluation: EvaluationResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in evaluation.examples:
            record = asdict(example)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_plot(path: Path, evaluation: EvaluationResult) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    contexts = sorted({row.context_id for row in evaluation.summaries})
    plotted_values = []
    for context_id in contexts:
        rows = [
            row
            for row in evaluation.summaries
            if row.context_id == context_id and row.canonical_percentage is not None
        ]
        x_values = [row.length for row in rows]
        y_values = [row.canonical_percentage for row in rows]
        plotted_values.extend(y_values)
        if len(contexts) == 1:
            axis.plot(x_values, y_values, linewidth=2, label=context_id)
            axis.fill_between(
                x_values,
                [row.canonical_ci95_low for row in rows],
                [row.canonical_ci95_high for row in rows],
                alpha=0.15,
            )
        else:
            label = context_id if len(contexts) <= 6 else None
            axis.plot(
                x_values,
                y_values,
                linewidth=1,
                alpha=0.35,
                label=label,
            )

    if len(contexts) > 1:
        pooled_rows = [
            row
            for row in evaluation.pooled_summaries
            if row.canonical_percentage is not None
        ]
        pooled_x = [row.length for row in pooled_rows]
        pooled_y = [row.canonical_percentage for row in pooled_rows]
        plotted_values.extend(pooled_y)
        axis.plot(
            pooled_x,
            pooled_y,
            color="black",
            linewidth=2.5,
            label="pooled",
        )
        if all(row.canonical_ci95_low is not None for row in pooled_rows):
            axis.fill_between(
                pooled_x,
                [row.canonical_ci95_low for row in pooled_rows],
                [row.canonical_ci95_high for row in pooled_rows],
                color="black",
                alpha=0.12,
            )

    axis.set_xlabel("Sampled continuation tokens")
    axis.set_ylabel("Canonical sequences (%)")
    axis.grid(True, alpha=0.25)
    axis.set_ylim(max(0.0, min(plotted_values, default=100.0) - 5.0), 100.5)
    if len(contexts) > 1 and len(contexts) <= 6:
        axis.legend()
    elif len(contexts) > 6:
        axis.legend(handles=[axis.lines[-1]], labels=["pooled"])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
