import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from canonicity.cli import parse_lengths
from canonicity.matrix_cli import main as matrix_main


class LengthParsingTests(unittest.TestCase):
    def test_ranges_are_inclusive_and_deduplicated(self):
        self.assertEqual(parse_lengths("1:3,3,5"), (1, 2, 3, 5))

    def test_stepped_ranges_include_the_requested_endpoint(self):
        self.assertEqual(parse_lengths("4:12:4"), (4, 8, 12))
        self.assertEqual(parse_lengths("4:11:4"), (4, 8, 11))

    def test_invalid_range_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_lengths("4:2")

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_lengths("1:4:0")

    def test_matrix_rejects_zero_batch_size(self):
        with self.assertRaises(SystemExit):
            matrix_main(
                [
                    "--model",
                    "mamba-130m",
                    "--condition",
                    "unconditional",
                    "--batch-size",
                    "0",
                    "--dry-run",
                ]
            )

    def test_matrix_rejects_explicit_device_with_active_map(self):
        with self.assertRaises(SystemExit):
            matrix_main(
                [
                    "--model",
                    "qwen3-30b-a3b-instruct-2507",
                    "--condition",
                    "unconditional",
                    "--device",
                    "cuda:1",
                    "--dry-run",
                ]
            )

    def test_matrix_dry_run_prints_job_and_rollout_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                matrix_main(
                    [
                        "--model",
                        "mamba-130m",
                        "--condition",
                        "unconditional",
                        "--unconditional-rollouts",
                        "17",
                        "--output-root",
                        str(Path(directory) / "matrix"),
                        "--dry-run",
                    ]
                )

        output = stdout.getvalue()
        self.assertIn("Matrix job 1/1", output)
        self.assertIn("contexts/prompts=1", output)
        self.assertIn("rollouts_per_context=17", output)
        self.assertIn("total_rollouts=17", output)

    def test_matrix_mamba_backend_is_explicitly_not_applicable(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            matrix_main(
                [
                    "--model",
                    "mamba-130m",
                    "--condition",
                    "unconditional",
                    "--dry-run",
                ]
            )

        output = stdout.getvalue()
        self.assertIn("attention_implementation=not_applicable", output)
        self.assertIn(
            "--attention-implementation not_applicable",
            output,
        )

    def test_matrix_transformer_defaults_to_flash_attention_2(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            matrix_main(
                [
                    "--model",
                    "gemma3-4b-it",
                    "--condition",
                    "unconditional",
                    "--dry-run",
                ]
            )

        output = stdout.getvalue()
        self.assertIn("attention_implementation=flash_attention_2", output)
        self.assertIn(
            "--attention-implementation flash_attention_2",
            output,
        )

    def test_matrix_sdpa_overrides_transformer_default(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            matrix_main(
                [
                    "--model",
                    "llama2-7b",
                    "--condition",
                    "unconditional",
                    "--attention-implementation",
                    "sdpa",
                    "--dry-run",
                ]
            )

        output = stdout.getvalue()
        self.assertIn("attention_implementation=sdpa", output)
        self.assertIn("--attention-implementation sdpa", output)
        self.assertNotIn("--attention-implementation flash_attention_2", output)

    def test_matrix_attention_override_leaves_mamba_not_applicable(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            matrix_main(
                [
                    "--model",
                    "llama2-7b",
                    "--model",
                    "mamba-130m",
                    "--condition",
                    "unconditional",
                    "--attention-implementation",
                    "sdpa",
                    "--dry-run",
                ]
            )

        output = stdout.getvalue()
        self.assertIn("attention_implementation=sdpa", output)
        self.assertIn("attention_implementation=not_applicable", output)
        self.assertIn("--attention-implementation not_applicable", output)

    def test_matrix_rejects_attention_override_for_mamba_only(self):
        with self.assertRaisesRegex(
            SystemExit,
            "cannot override models without attention",
        ):
            matrix_main(
                [
                    "--model",
                    "mamba-130m",
                    "--condition",
                    "unconditional",
                    "--attention-implementation",
                    "sdpa",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
