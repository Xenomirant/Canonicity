"""Inspectable outputs for dense non-canonicity segment analysis."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .sample_store import plan_fingerprint
from .segments import SEGMENT_ANALYSIS_IMPLEMENTATION, SegmentAnalysis


def _clean_optional_values(row: dict[str, Any]) -> dict[str, Any]:
    return {key: "" if value is None else value for key, value in row.items()}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_clean_optional_values(row))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_segment_report(
    output_dir: Path,
    analysis: SegmentAnalysis,
    *,
    source_samples: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_fields = [
        "context_id",
        "sample_index",
        "generated_tokens",
        "termination",
        "noncanonical_prefixes",
        "segment_count",
        "first_segment_start",
        "last_segment_end",
        "longest_segment",
    ]
    _write_csv(
        output_dir / "rollout_segments.csv",
        (
            {
                key: value
                for key, value in asdict(rollout).items()
                if key != "segments"
            }
            for rollout in analysis.rollouts
        ),
        rollout_fields,
    )

    with (output_dir / "segments.jsonl").open("w", encoding="utf-8") as handle:
        for rollout in analysis.rollouts:
            handle.write(json.dumps(asdict(rollout), ensure_ascii=False) + "\n")

    recurrence_fields = list(asdict(analysis.recurrence[0]).keys()) if analysis.recurrence else []
    if recurrence_fields:
        _write_csv(
            output_dir / "recurrence.csv",
            (asdict(row) for row in analysis.recurrence),
            recurrence_fields,
        )

    counts_by_context: dict[str, Counter[int]] = defaultdict(Counter)
    pooled_counts: Counter[int] = Counter()
    for rollout in analysis.rollouts:
        counts_by_context[rollout.context_id][rollout.segment_count] += 1
        pooled_counts[rollout.segment_count] += 1
    distribution_rows = []
    for context_id, counter in sorted(counts_by_context.items()):
        total = sum(counter.values())
        for segment_count, rollouts in sorted(counter.items()):
            distribution_rows.append(
                {
                    "scope": "context",
                    "context_id": context_id,
                    "segment_count": segment_count,
                    "rollouts": rollouts,
                    "percentage": 100.0 * rollouts / total,
                }
            )
    pooled_total = sum(pooled_counts.values())
    for segment_count, rollouts in sorted(pooled_counts.items()):
        distribution_rows.append(
            {
                "scope": "pooled",
                "context_id": "",
                "segment_count": segment_count,
                "rollouts": rollouts,
                "percentage": 100.0 * rollouts / pooled_total,
            }
        )
    _write_csv(
        output_dir / "segment_count_distribution.csv",
        distribution_rows,
        ["scope", "context_id", "segment_count", "rollouts", "percentage"],
    )

    definitions = {
        "analysis_implementation": SEGMENT_ANALYSIS_IMPLEMENTATION,
        "state": (
            "prefix t is non-canonical iff sampled_ids[:t] != "
            "encode(decode(sampled_ids[:t]))"
        ),
        "segment": "maximal consecutive run of non-canonical prefix states",
        "recurrence_landmark": "horizon/2",
        "recurrence_risk_set": (
            "rollout reached the horizon and is canonical at the landmark; "
            "rollouts inside a non-canonical segment are excluded"
        ),
        "recurrence_exposure": (
            "at least one prior segment completed before the landmark"
        ),
        "recurrence_outcome": (
            "at least one new segment starts after the landmark through horizon"
        ),
        "survival_condition": (
            "rollout retained at least horizon observed continuation token IDs"
        ),
        "prompt_adjustment": (
            "exact conditional test stratified by context_id when multiple "
            "contexts are present; Fisher exact test for one context"
        ),
        "test_sidedness": (
            "two-sided probability ordering; direction is reported by the "
            "common odds ratio and pooled descriptive risk difference"
        ),
        "pooled_probability_columns": (
            "descriptive pooled probabilities and risk difference are unadjusted; "
            "the common odds ratio and exact test are prompt-stratified"
        ),
        "causal_interpretation": (
            "association only; a significant result does not show that one "
            "non-canonical segment causes another"
        ),
        "common_odds_ratio_interpretation": (
            "Mantel-Haenszel common-odds-ratio estimate; a single common effect "
            "is only a useful summary when prompt-specific odds ratios are "
            "sufficiently homogeneous"
        ),
        "raw_segment_count_interpretation": (
            "lifetime count over the observed continuation; comparisons can be "
            "confounded by EOS-driven differences in observed token exposure"
        ),
    }
    with (output_dir / "segment_definitions.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(definitions, handle, indent=2, sort_keys=True)
        handle.write("\n")

    source_path = source_samples or output_dir / "samples.jsonl"
    source_record = None
    if source_path.exists():
        try:
            displayed_source_path = str(source_path.relative_to(output_dir))
        except ValueError:
            displayed_source_path = str(source_path.resolve())
        source_record = {
            "path": displayed_source_path,
            "sha256": _sha256(source_path),
        }

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

    artifact_names = (
        "segments.jsonl",
        "rollout_segments.csv",
        "recurrence.csv",
        "segment_count_distribution.csv",
        "segment_definitions.json",
    )
    manifest = {
        "schema_version": 1,
        "analysis_implementation": SEGMENT_ANALYSIS_IMPLEMENTATION,
        "sampling_plan_fingerprint": sampling_plan_fingerprint,
        "source_samples": source_record,
        "rollouts": len(analysis.rollouts),
        "recurrence_horizons": [row.horizon for row in analysis.recurrence],
        "artifacts": {
            name: {"sha256": _sha256(output_dir / name)}
            for name in artifact_names
            if (output_dir / name).exists()
        },
    }
    with (output_dir / "segment_analysis_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
