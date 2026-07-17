import argparse
import unittest

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


if __name__ == "__main__":
    unittest.main()
