import csv
import tempfile
import unittest
from pathlib import Path

from canonicity.segment_reporting import write_segment_report
from canonicity.segments import RolloutSegmentResult, SegmentAnalysis


class SegmentReportingTests(unittest.TestCase):
    def test_pooled_distribution_cannot_collide_with_context_id(self):
        rollout = RolloutSegmentResult(
            context_id="__pooled__",
            sample_index=0,
            generated_tokens=4,
            termination="max_length",
            noncanonical_prefixes=0,
            segment_count=0,
            first_segment_start=None,
            last_segment_end=None,
            longest_segment=0,
            segments=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_segment_report(output, SegmentAnalysis((rollout,), ()))
            with (output / "segment_count_distribution.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["scope"], "context")
        self.assertEqual(rows[0]["context_id"], "__pooled__")
        self.assertEqual(rows[1]["scope"], "pooled")
        self.assertEqual(rows[1]["context_id"], "")


if __name__ == "__main__":
    unittest.main()
