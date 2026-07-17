import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from canonicity.aggregate_cli import (
    _expected_summaries,
    _sha256,
    _validate_attention_provenance,
    main,
)
from canonicity.core import CANONICITY_EVALUATION_IMPLEMENTATION
from canonicity.generation import SAMPLING_IMPLEMENTATION
from canonicity.matrix_cli import (
    MATRIX_CHECKPOINTS,
    MATRIX_CONDITIONS,
    MATRIX_RECURRENCE_HORIZONS,
    MODEL_SPECS,
)
from canonicity.sample_store import plan_fingerprint
from canonicity.segment_reporting import write_segment_report
from canonicity.segments import (
    NoncanonicalSegment,
    RolloutSegmentResult,
    SegmentAnalysis,
    recurrence_rows,
)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_dataclass_csv(path, rows):
    records = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {name: "" if value is None else value for name, value in record.items()}
            )


def make_rollout(context_id, sample_index):
    patterns = {
        0: (10, 70, 200, 300, 600),
        1: (10,),
        2: (70, 200, 300, 600),
        3: (),
    }
    positions = patterns.get(sample_index, ())
    generated_tokens = 1024 if sample_index < 4 else 0
    segments = tuple(NoncanonicalSegment(value, value) for value in positions)
    return RolloutSegmentResult(
        context_id=context_id,
        sample_index=sample_index,
        generated_tokens=generated_tokens,
        termination="eos_token:2",
        noncanonical_prefixes=len(segments),
        segment_count=len(segments),
        first_segment_start=segments[0].start if segments else None,
        last_segment_end=segments[-1].end if segments else None,
        longest_segment=1 if segments else 0,
        segments=segments,
    )


def make_matrix(root, attention_overrides=None):
    attention_overrides = attention_overrides or {}
    prompted = [
        {"id": f"prompt-{index:03d}", "text": f"text {index}", "metadata": {}}
        for index in range(100)
    ]
    for model_position, (alias, spec) in enumerate(MODEL_SPECS.items()):
        for condition in MATRIX_CONDITIONS:
            job = root / alias / condition
            job.mkdir(parents=True)
            prompts = (
                [{"id": "unconditional", "text": None, "metadata": {}}]
                if condition == "unconditional"
                else prompted
            )
            samples_per_context = 32 if condition == "unconditional" else 64
            attention_implementation = attention_overrides.get(
                alias,
                spec.default_attention_implementation,
            )
            if attention_implementation == "flash_attention_2":
                attention_provider = "flash-attn"
                attention_provider_version = "test-flash-attn"
            elif attention_implementation == "sdpa":
                attention_provider = "torch"
                attention_provider_version = "test-torch"
            else:
                attention_provider = "not_applicable"
                attention_provider_version = None
            plan = {
                "sampling_implementation": SAMPLING_IMPLEMENTATION,
                "transformers_version": "test-transformers",
                "torch_version": "test-torch",
                "tokenizers_version": "test-tokenizers",
                "accelerate_version": "test-accelerate",
                "model_id": spec.model_id,
                "model_class": f"{alias}-class",
                "tokenizer_id": spec.model_id,
                "tokenizer_class": f"{alias}-tokenizer-class",
                "tokenizer_is_fast": True,
                "requested_model_revision": None,
                "requested_tokenizer_revision": None,
                "model_commit": f"{alias}-commit",
                "tokenizer_commit": f"{alias}-commit",
                "resolved_parameter_dtypes": ["test-dtype"],
                "attention_implementation": attention_implementation,
                "attention_provider": attention_provider,
                "attention_provider_version": attention_provider_version,
                "unconditional_seed_is_evaluated": False,
                "prompt_tokens_are_evaluated": False,
                "other_sampled_special_tokens_are_evaluated": True,
                "requested_dtype": "auto",
                "samples_per_context": samples_per_context,
                "max_new_tokens": 2048,
                "batch_size": spec.default_batch_size,
                "base_seed": 0,
                "seed_scheme": "sha256-batch-v1",
                "requested_device": "auto",
                "resolved_device": "cpu",
                "hardware_signature": {
                    "resolved_device": "cpu",
                    "machine": "test",
                    "processor": "test",
                },
                "requested_device_map": spec.default_device_map,
                "resolved_device_map": {},
                "eos_token_ids": [2],
                "prompt_mode": "raw",
                "sampling": {
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                },
                "prompts": prompts,
                "context_inputs": [
                    {
                        "id": prompt["id"],
                        "raw_prompt_tokens": (
                            0 if prompt["text"] is None else 2 + model_position
                        ),
                        "model_input_tokens": (
                            1 if prompt["text"] is None else 3 + model_position
                        ),
                    }
                    for prompt in prompts
                ],
            }
            fingerprint = plan_fingerprint(plan)
            write_json(
                job / "sampling_plan.json",
                {
                    "schema_version": 1,
                    "plan_fingerprint": fingerprint,
                    "plan": plan,
                },
            )

            rollouts = tuple(
                make_rollout(prompt["id"], sample_index)
                for prompt in prompts
                for sample_index in range(samples_per_context)
            )
            with (job / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for rollout in rollouts:
                    prefix_canonicity = {
                        str(length): not any(
                            segment.start <= length <= segment.end
                            for segment in rollout.segments
                        )
                        for length in MATRIX_CHECKPOINTS
                        if length <= rollout.generated_tokens
                    }
                    handle.write(
                        json.dumps(
                            {
                                "context_id": rollout.context_id,
                                "sample_index": rollout.sample_index,
                                "generated_token_ids": [1]
                                * rollout.generated_tokens,
                                "termination": rollout.termination,
                                "prefix_canonicity": prefix_canonicity,
                            }
                        )
                        + "\n"
                    )

            context_rows, pooled_rows = _expected_summaries(
                rollouts, [prompt["id"] for prompt in prompts]
            )
            write_dataclass_csv(job / "summary.csv", context_rows)
            if condition == "wikitext":
                write_dataclass_csv(job / "pooled_summary.csv", pooled_rows)
            (job / "noncanonical_examples.jsonl").write_text("", encoding="utf-8")
            metadata = {
                "model": spec.model_id,
                "tokenizer": spec.model_id,
                "lengths": list(MATRIX_CHECKPOINTS),
                "recurrence_horizons": list(MATRIX_RECURRENCE_HORIZONS),
                "samples_per_context": samples_per_context,
                "prompt_mode": "raw",
                "prompts": prompts,
                "context_inputs": plan["context_inputs"],
                "batch_size": plan["batch_size"],
                "seed": 0,
                "sampling": plan["sampling"],
                "requested_revision": None,
                "requested_tokenizer_revision": None,
                "canonical_reencoding_add_special_tokens": False,
                "decode_skip_special_tokens": False,
                "decode_clean_up_tokenization_spaces": False,
            }
            for name in (
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
            ):
                metadata[name] = plan[name]
            write_json(job / "metadata.json", metadata)

            evaluation_artifacts = [
                "metadata.json",
                "summary.csv",
                "samples.jsonl",
                "noncanonical_examples.jsonl",
            ]
            if condition == "wikitext":
                evaluation_artifacts.append("pooled_summary.csv")
            write_json(
                job / "evaluation_manifest.json",
                {
                    "schema_version": 1,
                    "evaluation_implementation": CANONICITY_EVALUATION_IMPLEMENTATION,
                    "sampling_plan_fingerprint": fingerprint,
                    "rollouts": len(rollouts),
                    "artifacts": {
                        name: {"sha256": _sha256(job / name)}
                        for name in evaluation_artifacts
                    },
                },
            )
            analysis = SegmentAnalysis(
                rollouts=rollouts,
                recurrence=recurrence_rows(rollouts, MATRIX_RECURRENCE_HORIZONS),
            )
            write_segment_report(job, analysis, source_samples=job / "samples.jsonl")


class AggregateTests(unittest.TestCase):
    def test_attention_provenance_accepts_each_declared_backend(self):
        cases = (
            (
                "gemma3-4b-it",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "flash_attention_2",
                    "attention_provider": "flash-attn",
                    "attention_provider_version": "2.8.3",
                },
            ),
            (
                "gemma3-4b-it",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "sdpa",
                    "attention_provider": "torch",
                    "attention_provider_version": "2.10.0",
                },
            ),
            (
                "mamba-130m",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "not_applicable",
                    "attention_provider": "not_applicable",
                    "attention_provider_version": None,
                },
            ),
        )

        for alias, plan in cases:
            with self.subTest(alias=alias, backend=plan["attention_implementation"]):
                _validate_attention_provenance(plan, alias, Path("job"))

    def test_attention_provenance_rejects_inconsistent_combinations(self):
        cases = (
            (
                "gemma3-4b-it",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "flash_attention_2",
                    "attention_provider": "torch",
                    "attention_provider_version": "2.10.0",
                },
            ),
            (
                "gemma3-4b-it",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "flash_attention_2",
                    "attention_provider": "flash-attn",
                    "attention_provider_version": " ",
                },
            ),
            (
                "gemma3-4b-it",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "sdpa",
                    "attention_provider": "torch",
                    "attention_provider_version": "2.9.0",
                },
            ),
            (
                "gemma3-4b-it",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "not_applicable",
                    "attention_provider": "not_applicable",
                    "attention_provider_version": None,
                },
            ),
            (
                "mamba-130m",
                {
                    "torch_version": "2.10.0",
                    "attention_implementation": "sdpa",
                    "attention_provider": "torch",
                    "attention_provider_version": "2.10.0",
                },
            ),
        )

        for alias, plan in cases:
            with self.subTest(alias=alias, plan=plan):
                with self.assertRaises(ValueError):
                    _validate_attention_provenance(plan, alias, Path("job"))

    def test_complete_family_is_recomputed_and_log_adjusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "matrix"
            make_matrix(root)

            main(["--results-root", str(root)])

            with (root / "recurrence_all.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            metadata = json.loads(
                (root / "recurrence_all.metadata.json").read_text(encoding="utf-8")
            )
            with (root / "canonicity_all.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                canonicity_rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 50)
        self.assertEqual(metadata["planned_hypotheses"], 50)
        self.assertEqual(len(canonicity_rows), 70)
        self.assertEqual(metadata["untestable_hypotheses_assigned_p_one"], 10)
        self.assertTrue(
            all(row["multiplicity_p_value"] == "1.0" for row in rows[4::5])
        )
        self.assertTrue(all(row["log10_by_q_value"] for row in rows))
        self.assertEqual(len(metadata["source_integrity"]), 10)

    def test_incomplete_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "matrix"
            make_matrix(root)
            (root / "mamba-130m" / "wikitext" / "recurrence.csv").unlink()

            with self.assertRaises(SystemExit):
                main(["--results-root", str(root)])

    def test_mixed_transformer_attention_backends_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "matrix"
            make_matrix(root, {"llama2-7b": "sdpa"})

            with self.assertRaisesRegex(
                ValueError,
                "attention backend changed across Transformer models",
            ):
                main(["--results-root", str(root)])

    def test_modified_recurrence_is_rejected_even_if_schema_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "matrix"
            make_matrix(root)
            path = root / "mamba-130m" / "unconditional" / "recurrence.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["p_value"] = "0.123"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = path.with_name("segment_analysis_manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["recurrence.csv"]["sha256"] = _sha256(path)
            write_json(manifest_path, manifest)

            with self.assertRaises(ValueError):
                main(["--results-root", str(root)])


if __name__ == "__main__":
    unittest.main()
