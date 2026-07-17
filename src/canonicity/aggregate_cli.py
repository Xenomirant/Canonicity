"""Validate and aggregate the complete planned recurrence-test family."""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .core import (
    CANONICITY_EVALUATION_IMPLEMENTATION,
    SummaryRow,
    _wilson_percentage_interval,
)
from .generation import SAMPLING_IMPLEMENTATION
from .matrix_cli import (
    MATRIX_CHECKPOINTS,
    MATRIX_CONDITIONS,
    MATRIX_RECURRENCE_HORIZONS,
    MODEL_SPECS,
)
from .multiple_testing import benjamini_yekutieli_log
from .sample_store import plan_fingerprint
from .segments import (
    SEGMENT_ANALYSIS_IMPLEMENTATION,
    NoncanonicalSegment,
    RolloutSegmentResult,
    recurrence_rows,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all five models, both conditions, and every planned "
            "horizon, then apply a dependence-robust Benjamini-Yekutieli "
            "false-discovery-rate correction."
        )
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to RESULTS_ROOT/recurrence_all.csv",
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_paths(results_root: Path) -> Dict[Tuple[str, str], Path]:
    return {
        (alias, condition): results_root / alias / condition
        for alias in MODEL_SPECS
        for condition in MATRIX_CONDITIONS
    }


def _validate_attention_provenance(
    plan: Dict[str, Any],
    alias: str,
    path: Path,
) -> None:
    """Require one internally consistent, model-applicable attention backend."""

    implementation = plan.get("attention_implementation")
    provider = plan.get("attention_provider")
    provider_version = plan.get("attention_provider_version")
    attention_is_applicable = (
        MODEL_SPECS[alias].default_attention_implementation != "not_applicable"
    )

    valid = False
    if not attention_is_applicable:
        valid = (
            implementation == "not_applicable"
            and provider == "not_applicable"
            and provider_version is None
        )
    elif implementation == "flash_attention_2":
        valid = (
            provider == "flash-attn"
            and isinstance(provider_version, str)
            and bool(provider_version.strip())
        )
    elif implementation == "sdpa":
        valid = (
            provider == "torch"
            and isinstance(provider_version, str)
            and bool(provider_version.strip())
            and provider_version == plan.get("torch_version")
        )

    if not valid:
        raise ValueError(
            f"invalid attention backend provenance for {alias}: {path}"
        )


def _validate_plan(
    path: Path,
    alias: str,
    condition: str,
) -> Tuple[Dict[str, Any], str]:
    manifest = _load_json(path / "sampling_plan.json")
    plan = manifest.get("plan")
    fingerprint = manifest.get("plan_fingerprint")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(plan, dict)
        or fingerprint != plan_fingerprint(plan)
    ):
        raise ValueError(f"invalid sampling plan fingerprint: {path}")

    spec = MODEL_SPECS[alias]
    if plan.get("model_id") != spec.model_id:
        raise ValueError(f"wrong model id for {alias}: {path}")
    if plan.get("tokenizer_id") != spec.model_id:
        raise ValueError(f"matrix jobs must use each model's tokenizer: {path}")
    if not all(
        isinstance(plan.get(name), str) and plan[name]
        for name in (
            "model_commit",
            "tokenizer_commit",
            "model_class",
            "tokenizer_class",
            "transformers_version",
            "torch_version",
            "tokenizers_version",
            "accelerate_version",
        )
    ):
        raise ValueError(f"matrix jobs require resolved runtime provenance: {path}")
    if plan["model_commit"] != plan["tokenizer_commit"]:
        raise ValueError(f"same-repository model/tokenizer commits differ: {path}")
    _validate_attention_provenance(plan, alias, path)

    prompts = plan.get("prompts")
    expected_prompt_count = 1 if condition == "unconditional" else 100
    if not isinstance(prompts, list) or len(prompts) != expected_prompt_count:
        raise ValueError(
            f"{condition} requires {expected_prompt_count} contexts: {path}"
        )
    if any(
        not isinstance(prompt, dict)
        or not isinstance(prompt.get("id"), str)
        or not prompt["id"]
        for prompt in prompts
    ):
        raise ValueError(f"invalid prompt records: {path}")
    prompt_ids = [prompt["id"] for prompt in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError(f"duplicate prompt ids: {path}")
    if condition == "unconditional":
        if prompts[0].get("text") is not None:
            raise ValueError(f"unconditional job contains prompt text: {path}")
    elif any(not isinstance(prompt.get("text"), str) for prompt in prompts):
        raise ValueError(f"conditioned job contains a non-text prompt: {path}")
    context_inputs = plan.get("context_inputs")
    if (
        not isinstance(context_inputs, list)
        or len(context_inputs) != len(prompts)
        or [record.get("id") for record in context_inputs] != prompt_ids
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("raw_prompt_tokens"), int)
            or isinstance(record.get("raw_prompt_tokens"), bool)
            or record["raw_prompt_tokens"] < 0
            or not isinstance(record.get("model_input_tokens"), int)
            or isinstance(record.get("model_input_tokens"), bool)
            or record["model_input_tokens"] < 1
            for record in context_inputs
        )
    ):
        raise ValueError(f"invalid preflight context inputs: {path}")

    if (
        not isinstance(plan.get("samples_per_context"), int)
        or isinstance(plan.get("samples_per_context"), bool)
        or plan["samples_per_context"] < 1
    ):
        raise ValueError(f"samples_per_context must be positive: {path}")
    if plan.get("max_new_tokens") != max(MATRIX_CHECKPOINTS):
        raise ValueError(f"wrong continuation horizon: {path}")
    if plan.get("requested_dtype") != "auto":
        raise ValueError(f"primary matrix must preserve checkpoint-native dtype: {path}")
    if not isinstance(plan.get("resolved_parameter_dtypes"), list) or not plan[
        "resolved_parameter_dtypes"
    ]:
        raise ValueError(f"missing resolved parameter dtypes: {path}")
    if (
        plan.get("sampling_implementation") != SAMPLING_IMPLEMENTATION
        or plan.get("unconditional_seed_is_evaluated") is not False
        or plan.get("prompt_tokens_are_evaluated") is not False
        or plan.get("other_sampled_special_tokens_are_evaluated") is not True
        or plan.get("seed_scheme") != "sha256-batch-v1"
        or plan.get("prompt_mode") != "raw"
        or plan.get("sampling")
        != {
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        }
    ):
        raise ValueError(f"job uses a different sampling estimand: {path}")
    if not isinstance(plan.get("eos_token_ids"), list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in plan["eos_token_ids"]
    ):
        raise ValueError(f"invalid EOS policy: {path}")
    if not isinstance(plan.get("hardware_signature"), dict):
        raise ValueError(f"missing hardware signature: {path}")
    return plan, str(fingerprint)


def _validate_artifact_manifest(
    job_path: Path,
    filename: str,
    implementation_field: str,
    implementation: str,
    plan_fingerprint_value: str,
    expected_artifacts: Iterable[str],
) -> Dict[str, Any]:
    path = job_path / filename
    manifest = _load_json(path)
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != 1
        or manifest.get(implementation_field) != implementation
        or manifest.get("sampling_plan_fingerprint") != plan_fingerprint_value
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(expected_artifacts)
    ):
        raise ValueError(f"invalid or stale artifact manifest: {path}")
    for artifact_name, record in artifacts.items():
        artifact_path = job_path / artifact_name
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256"}
            or record["sha256"] != _sha256(artifact_path)
        ):
            raise ValueError(f"artifact hash mismatch: {artifact_path}")
    return manifest


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _nonblank_json_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                yield line_number, value


def _validate_termination(
    termination: str,
    generated_tokens: int,
    plan: Dict[str, Any],
    label: str,
) -> None:
    maximum = plan["max_new_tokens"]
    if generated_tokens > maximum:
        raise ValueError(f"generated continuation exceeds the plan: {label}")
    if termination == "max_length":
        if generated_tokens != maximum:
            raise ValueError(f"short max_length rollout: {label}")
        return
    prefix = "eos_token:"
    if not termination.startswith(prefix):
        raise ValueError(f"invalid termination reason: {label}")
    try:
        eos_id = int(termination[len(prefix) :])
    except ValueError as error:
        raise ValueError(f"invalid EOS termination: {label}") from error
    if eos_id not in plan["eos_token_ids"] or generated_tokens >= maximum:
        raise ValueError(f"EOS termination does not match the plan: {label}")


def _load_validated_rollouts(
    job_path: Path,
    plan: Dict[str, Any],
) -> Tuple[RolloutSegmentResult, ...]:
    expected_keys = [
        (prompt["id"], sample_index)
        for prompt in plan["prompts"]
        for sample_index in range(plan["samples_per_context"])
    ]
    sample_iterator = _nonblank_json_lines(job_path / "samples.jsonl")
    segment_iterator = _nonblank_json_lines(job_path / "segments.jsonl")
    rollouts = []
    sample_fields = {
        "context_id",
        "sample_index",
        "generated_token_ids",
        "termination",
        "prefix_canonicity",
    }
    rollout_fields = {field.name for field in fields(RolloutSegmentResult)}
    segment_fields = {"start", "end"}

    for expected_key in expected_keys:
        try:
            sample_line, sample = next(sample_iterator)
            segment_line, raw_rollout = next(segment_iterator)
        except StopIteration as error:
            raise ValueError(f"incomplete sample/segment artifacts: {job_path}") from error
        sample_label = f"{job_path / 'samples.jsonl'}:{sample_line}"
        segment_label = f"{job_path / 'segments.jsonl'}:{segment_line}"
        if set(sample) != sample_fields:
            raise ValueError(f"wrong sample schema: {sample_label}")
        if set(raw_rollout) != rollout_fields:
            raise ValueError(f"wrong segment-rollout schema: {segment_label}")

        sample_key = (
            sample["context_id"],
            _integer(sample["sample_index"], f"sample index at {sample_label}"),
        )
        rollout_key = (
            raw_rollout["context_id"],
            _integer(
                raw_rollout["sample_index"],
                f"sample index at {segment_label}",
            ),
        )
        if sample_key != expected_key or rollout_key != expected_key:
            raise ValueError(f"unexpected rollout order or identity: {job_path}")
        if not isinstance(sample["generated_token_ids"], list) or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in sample["generated_token_ids"]
        ):
            raise ValueError(f"invalid generated token ids: {sample_label}")
        generated_tokens = len(sample["generated_token_ids"])
        if raw_rollout["generated_tokens"] != generated_tokens:
            raise ValueError(f"sample/segment length mismatch: {segment_label}")
        termination = sample["termination"]
        if not isinstance(termination, str) or raw_rollout["termination"] != termination:
            raise ValueError(f"sample/segment termination mismatch: {segment_label}")
        _validate_termination(termination, generated_tokens, plan, sample_label)

        raw_segments = raw_rollout["segments"]
        if not isinstance(raw_segments, list):
            raise ValueError(f"segments must be a list: {segment_label}")
        segments = []
        previous_end = -1
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict) or set(raw_segment) != segment_fields:
                raise ValueError(f"invalid segment schema: {segment_label}")
            start = _integer(raw_segment["start"], f"segment start at {segment_label}", 1)
            end = _integer(raw_segment["end"], f"segment end at {segment_label}", 1)
            if start > end or end > generated_tokens or start <= previous_end + 1:
                raise ValueError(f"segments are not disjoint maximal runs: {segment_label}")
            segments.append(NoncanonicalSegment(start, end))
            previous_end = end
        segments_tuple = tuple(segments)
        count = len(segments_tuple)
        noncanonical_prefixes = sum(segment.length for segment in segments_tuple)
        expected_derived = {
            "segment_count": count,
            "noncanonical_prefixes": noncanonical_prefixes,
            "first_segment_start": segments_tuple[0].start if count else None,
            "last_segment_end": segments_tuple[-1].end if count else None,
            "longest_segment": max(
                (segment.length for segment in segments_tuple), default=0
            ),
        }
        if any(raw_rollout[name] != value for name, value in expected_derived.items()):
            raise ValueError(f"inconsistent segment summaries: {segment_label}")

        prefix_canonicity = sample["prefix_canonicity"]
        if not isinstance(prefix_canonicity, dict):
            raise ValueError(f"prefix_canonicity must be an object: {sample_label}")
        expected_prefixes = {
            str(length) for length in MATRIX_CHECKPOINTS if length <= generated_tokens
        }
        if set(prefix_canonicity) != expected_prefixes or any(
            not isinstance(value, bool) for value in prefix_canonicity.values()
        ):
            raise ValueError(f"wrong evaluated prefix set: {sample_label}")
        for raw_length, is_canonical in prefix_canonicity.items():
            length = int(raw_length)
            expected_canonical = not any(
                segment.start <= length <= segment.end
                for segment in segments_tuple
            )
            if is_canonical is not expected_canonical:
                raise ValueError(f"sample/segment state mismatch: {sample_label}")

        rollouts.append(
            RolloutSegmentResult(
                context_id=expected_key[0],
                sample_index=expected_key[1],
                generated_tokens=generated_tokens,
                termination=termination,
                noncanonical_prefixes=noncanonical_prefixes,
                segment_count=count,
                first_segment_start=expected_derived["first_segment_start"],
                last_segment_end=expected_derived["last_segment_end"],
                longest_segment=expected_derived["longest_segment"],
                segments=segments_tuple,
            )
        )

    try:
        next(sample_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"extra samples beyond the plan: {job_path}")
    try:
        next(segment_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"extra segment rollouts beyond the plan: {job_path}")
    return tuple(rollouts)


def _values_match(raw: str, expected: Any) -> bool:
    if expected is None:
        return raw == ""
    if isinstance(expected, int):
        try:
            return int(raw) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            observed = float(raw)
        except ValueError:
            return False
        if math.isnan(observed):
            return False
        if math.isinf(expected):
            return observed == expected
        return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-300)
    return raw == str(expected)


def _validate_csv_rows(path: Path, expected_rows: Sequence[Any]) -> None:
    if not expected_rows:
        raise ValueError(f"cannot validate an empty expected CSV: {path}")
    expected_records = [asdict(row) for row in expected_rows]
    fieldnames = list(expected_records[0])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"wrong CSV schema: {path}")
        rows = list(reader)
    if len(rows) != len(expected_records):
        raise ValueError(f"wrong CSV row count: {path}")
    for row_number, (row, expected) in enumerate(
        zip(rows, expected_records), start=2
    ):
        for name, value in expected.items():
            if not _values_match(row[name], value):
                raise ValueError(
                    f"derived CSV disagrees with source artifacts at "
                    f"{path}:{row_number}:{name}"
                )


def _expected_summaries(
    rollouts: Sequence[RolloutSegmentResult],
    context_ids: Sequence[str],
) -> Tuple[Tuple[SummaryRow, ...], Tuple[SummaryRow, ...]]:
    by_context = []
    for context_id in sorted(context_ids):
        context_rollouts = [
            rollout for rollout in rollouts if rollout.context_id == context_id
        ]
        total = len(context_rollouts)
        for length in MATRIX_CHECKPOINTS:
            eligible = [
                rollout
                for rollout in context_rollouts
                if rollout.generated_tokens >= length
            ]
            canonical = sum(
                not any(
                    segment.start <= length <= segment.end
                    for segment in rollout.segments
                )
                for rollout in eligible
            )
            denominator = len(eligible)
            percentage = 100.0 * canonical / denominator if denominator else None
            ci_low, ci_high = _wilson_percentage_interval(canonical, denominator)
            by_context.append(
                SummaryRow(
                    context_id=context_id,
                    length=length,
                    total_samples=total,
                    eligible_sequences=denominator,
                    terminated_before_length=total - denominator,
                    canonical_sequences=canonical,
                    noncanonical_sequences=denominator - canonical,
                    canonical_percentage=percentage,
                    canonical_ci95_low=ci_low,
                    canonical_ci95_high=ci_high,
                )
            )

    pooled = []
    for length in MATRIX_CHECKPOINTS:
        rows = [row for row in by_context if row.length == length]
        total = sum(row.total_samples for row in rows)
        denominator = sum(row.eligible_sequences for row in rows)
        canonical = sum(row.canonical_sequences for row in rows)
        pooled.append(
            SummaryRow(
                context_id="__pooled__",
                length=length,
                total_samples=total,
                eligible_sequences=denominator,
                terminated_before_length=total - denominator,
                canonical_sequences=canonical,
                noncanonical_sequences=denominator - canonical,
                canonical_percentage=(
                    100.0 * canonical / denominator if denominator else None
                ),
                canonical_ci95_low=None,
                canonical_ci95_high=None,
            )
        )
    return tuple(by_context), tuple(pooled)


def _clean_record(value: Any) -> Dict[str, Any]:
    return {
        name: "" if field_value is None else field_value
        for name, field_value in asdict(value).items()
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    output = args.output or args.results_root / "recurrence_all.csv"
    jobs = _job_paths(args.results_root)
    expected_recurrence_paths = {path / "recurrence.csv" for path in jobs.values()}
    available_recurrence_paths = set(args.results_root.glob("*/*/recurrence.csv"))
    missing = sorted(expected_recurrence_paths - available_recurrence_paths)
    unexpected = sorted(available_recurrence_paths - expected_recurrence_paths)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(str(path) for path in missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(str(path) for path in unexpected))
        raise SystemExit(
            "recurrence aggregation requires the complete declared 10-job matrix; "
            + "; ".join(details)
        )

    records = []
    canonicity_records = []
    multiplicity_log_p_values = []
    condition_signatures: Dict[str, str] = {}
    condition_designs: Dict[str, Dict[str, int]] = {}
    model_revision_signatures: Dict[str, str] = {}
    plan_fingerprints: Dict[str, str] = {}
    source_integrity: Dict[str, Any] = {}
    global_runtime_signature = None
    global_attention_signature = None
    segment_definition_fingerprint = None
    observed_p_values = 0

    for (alias, condition), job_path in jobs.items():
        job_key = f"{alias}/{condition}"
        plan, sampling_fingerprint = _validate_plan(job_path, alias, condition)
        plan_fingerprints[job_key] = sampling_fingerprint

        evaluation_artifacts = {
            "metadata.json",
            "summary.csv",
            "samples.jsonl",
            "noncanonical_examples.jsonl",
        }
        if condition == "wikitext":
            evaluation_artifacts.add("pooled_summary.csv")
        evaluation_manifest = _validate_artifact_manifest(
            job_path,
            "evaluation_manifest.json",
            "evaluation_implementation",
            CANONICITY_EVALUATION_IMPLEMENTATION,
            sampling_fingerprint,
            evaluation_artifacts,
        )
        analysis_manifest = _validate_artifact_manifest(
            job_path,
            "segment_analysis_manifest.json",
            "analysis_implementation",
            SEGMENT_ANALYSIS_IMPLEMENTATION,
            sampling_fingerprint,
            {
                "segments.jsonl",
                "rollout_segments.csv",
                "recurrence.csv",
                "segment_count_distribution.csv",
                "segment_definitions.json",
            },
        )
        expected_rollouts = plan["samples_per_context"] * len(plan["prompts"])
        source_samples = analysis_manifest.get("source_samples")
        if (
            evaluation_manifest.get("rollouts") != expected_rollouts
            or analysis_manifest.get("rollouts") != expected_rollouts
            or analysis_manifest.get("recurrence_horizons")
            != list(MATRIX_RECURRENCE_HORIZONS)
            or not isinstance(source_samples, dict)
            or source_samples.get("path") != "samples.jsonl"
            or source_samples.get("sha256")
            != evaluation_manifest["artifacts"]["samples.jsonl"]["sha256"]
        ):
            raise ValueError(f"analysis is not linked to the sampled rollouts: {job_path}")

        runtime_signature = _json_fingerprint(
            {
                name: plan.get(name)
                for name in (
                    "sampling_implementation",
                    "transformers_version",
                    "torch_version",
                    "tokenizers_version",
                    "accelerate_version",
                )
            }
        )
        if global_runtime_signature is None:
            global_runtime_signature = runtime_signature
        elif global_runtime_signature != runtime_signature:
            raise ValueError(f"software runtime changed within the matrix: {job_path}")

        if plan["attention_implementation"] != "not_applicable":
            attention_signature = _json_fingerprint(
                {
                    name: plan.get(name)
                    for name in (
                        "attention_implementation",
                        "attention_provider",
                        "attention_provider_version",
                    )
                }
            )
            if global_attention_signature is None:
                global_attention_signature = attention_signature
            elif global_attention_signature != attention_signature:
                raise ValueError(
                    "attention backend changed across Transformer models: "
                    f"{job_path}"
                )

        model_revision_signature = _json_fingerprint(
            {
                name: plan.get(name)
                for name in (
                    "model_commit",
                    "tokenizer_commit",
                    "model_class",
                    "tokenizer_class",
                    "tokenizer_is_fast",
                    "resolved_parameter_dtypes",
                    "attention_implementation",
                    "attention_provider",
                    "attention_provider_version",
                    "requested_device",
                    "resolved_device",
                    "hardware_signature",
                    "requested_device_map",
                    "resolved_device_map",
                    "batch_size",
                    "eos_token_ids",
                )
            }
        )
        previous_model_revision = model_revision_signatures.setdefault(
            alias, model_revision_signature
        )
        if previous_model_revision != model_revision_signature:
            raise ValueError(
                f"model/tokenizer runtime changed between conditions: {job_path}"
            )
        condition_signature = _json_fingerprint(
            {
                name: plan.get(name)
                for name in (
                    "samples_per_context",
                    "max_new_tokens",
                    "base_seed",
                    "seed_scheme",
                    "prompt_mode",
                    "sampling",
                    "prompts",
                )
            }
        )
        previous_signature = condition_signatures.setdefault(
            condition, condition_signature
        )
        if previous_signature != condition_signature:
            raise ValueError(
                f"sampling plans are not model-paired within {condition}: {job_path}"
            )
        condition_designs.setdefault(
            condition,
            {
                "contexts": len(plan["prompts"]),
                "samples_per_context": plan["samples_per_context"],
            },
        )

        metadata = _load_json(job_path / "metadata.json")
        metadata_matches = {
            "model": MODEL_SPECS[alias].model_id,
            "tokenizer": MODEL_SPECS[alias].model_id,
            "lengths": list(MATRIX_CHECKPOINTS),
            "recurrence_horizons": list(MATRIX_RECURRENCE_HORIZONS),
            "samples_per_context": plan["samples_per_context"],
            "prompt_mode": plan["prompt_mode"],
            "prompts": plan["prompts"],
            "context_inputs": plan["context_inputs"],
            "batch_size": plan["batch_size"],
            "seed": plan["base_seed"],
            "sampling": plan["sampling"],
            "requested_revision": plan.get("requested_model_revision"),
            "requested_tokenizer_revision": plan.get(
                "requested_tokenizer_revision"
            ),
            "canonical_reencoding_add_special_tokens": False,
            "decode_skip_special_tokens": False,
            "decode_clean_up_tokenization_spaces": False,
        }
        runtime_metadata_fields = (
            "sampling_implementation",
            "transformers_version",
            "torch_version",
            "tokenizers_version",
            "accelerate_version",
            "model_commit",
            "tokenizer_commit",
            "model_class",
            "tokenizer_class",
            "tokenizer_is_fast",
            "resolved_parameter_dtypes",
            "attention_implementation",
            "attention_provider",
            "attention_provider_version",
            "requested_device",
            "resolved_device",
            "hardware_signature",
            "requested_device_map",
            "resolved_device_map",
            "eos_token_ids",
            "unconditional_seed_is_evaluated",
            "prompt_tokens_are_evaluated",
            "other_sampled_special_tokens_are_evaluated",
            "seed_scheme",
            "requested_dtype",
        )
        metadata_matches.update(
            {name: plan.get(name) for name in runtime_metadata_fields}
        )
        if any(metadata.get(name) != value for name, value in metadata_matches.items()):
            raise ValueError(f"job metadata does not match its sampling plan: {job_path}")

        definitions = _load_json(job_path / "segment_definitions.json")
        if definitions.get("analysis_implementation") != SEGMENT_ANALYSIS_IMPLEMENTATION:
            raise ValueError(f"wrong segment definition version: {job_path}")
        current_definition_fingerprint = _json_fingerprint(definitions)
        if segment_definition_fingerprint is None:
            segment_definition_fingerprint = current_definition_fingerprint
        elif segment_definition_fingerprint != current_definition_fingerprint:
            raise ValueError(f"mixed recurrence estimands: {job_path}")

        rollouts = _load_validated_rollouts(job_path, plan)
        context_summaries, pooled_summaries = _expected_summaries(
            rollouts, [prompt["id"] for prompt in plan["prompts"]]
        )
        _validate_csv_rows(job_path / "summary.csv", context_summaries)
        if condition == "wikitext":
            _validate_csv_rows(job_path / "pooled_summary.csv", pooled_summaries)
            selected_summaries = pooled_summaries
        else:
            selected_summaries = context_summaries
        canonicity_records.extend(
            {
                "model_alias": alias,
                "condition": condition,
                **_clean_record(row),
            }
            for row in selected_summaries
        )

        recomputed_recurrence = recurrence_rows(
            rollouts, MATRIX_RECURRENCE_HORIZONS
        )
        _validate_csv_rows(job_path / "recurrence.csv", recomputed_recurrence)
        for row in recomputed_recurrence:
            if row.p_value is None:
                multiplicity_p_value = 1.0
                multiplicity_log10_p_value = 0.0
            else:
                if row.log10_p_value is None or row.log10_p_value > 0:
                    raise ValueError(f"invalid p-value/log-p pair: {job_path}")
                if row.p_value > 0 and not math.isclose(
                    math.log10(row.p_value),
                    row.log10_p_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"inconsistent p-value/log-p pair: {job_path}")
                observed_p_values += 1
                multiplicity_p_value = row.p_value
                multiplicity_log10_p_value = row.log10_p_value
            multiplicity_log_p_values.append(
                multiplicity_log10_p_value * math.log(10.0)
            )
            records.append(
                {
                    "model_alias": alias,
                    "condition": condition,
                    **_clean_record(row),
                    "multiplicity_p_value": multiplicity_p_value,
                    "multiplicity_log10_p_value": multiplicity_log10_p_value,
                    "by_q_value": "",
                    "log10_by_q_value": "",
                }
            )

        source_integrity[job_key] = {
            "sampling_plan_fingerprint": sampling_fingerprint,
            "evaluation_manifest_sha256": _sha256(
                job_path / "evaluation_manifest.json"
            ),
            "segment_analysis_manifest_sha256": _sha256(
                job_path / "segment_analysis_manifest.json"
            ),
            "samples_sha256": source_samples["sha256"],
            "segments_sha256": analysis_manifest["artifacts"]["segments.jsonl"][
                "sha256"
            ],
            "recurrence_sha256": analysis_manifest["artifacts"]["recurrence.csv"][
                "sha256"
            ],
        }

    adjusted_logs = benjamini_yekutieli_log(multiplicity_log_p_values)
    for record, log_q_value in zip(records, adjusted_logs):
        record["by_q_value"] = math.exp(log_q_value)
        record["log10_by_q_value"] = log_q_value / math.log(10.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    canonicity_output = output.parent / "canonicity_all.csv"
    with canonicity_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(canonicity_records[0]))
        writer.writeheader()
        writer.writerows(canonicity_records)

    metadata = {
        "planned_hypotheses": len(records),
        "observed_p_values": observed_p_values,
        "untestable_hypotheses_assigned_p_one": len(records) - observed_p_values,
        "correction": (
            "Benjamini-Yekutieli false-discovery-rate adjustment under arbitrary "
            "dependence, computed in log space"
        ),
        "family": "5 models x 2 conditions x 5 fixed horizons",
        "segment_definition_fingerprint": segment_definition_fingerprint,
        "global_runtime_fingerprint": global_runtime_signature,
        "condition_design_fingerprints": condition_signatures,
        "condition_designs": condition_designs,
        "model_runtime_fingerprints": model_revision_signatures,
        "sampling_plan_fingerprints": plan_fingerprints,
        "source_integrity": source_integrity,
        "canonicity_summary": str(canonicity_output),
    }
    metadata_path = output.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(records)} validated recurrence rows to {output}")


if __name__ == "__main__":
    main()
