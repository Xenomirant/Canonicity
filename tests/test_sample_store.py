import tempfile
import unittest
from pathlib import Path

from canonicity.core import SampledSequence
from canonicity.sample_store import SampleBatchStore


class SampleBatchStoreTests(unittest.TestCase):
    def test_completed_batch_is_exactly_resumable(self):
        plan = {"model": "example/model", "seed": 7}
        samples = (
            SampledSequence("context", 0, (1, 2), "max_length"),
            SampledSequence("context", 1, (3,), "eos_token:0"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            store = SampleBatchStore(output, plan)
            store.write_batch(0, "context", 0, samples)

            resumed = SampleBatchStore(output, plan).load_batch(
                0, "context", 0, 2
            )

        self.assertEqual(resumed, samples)

    def test_different_plan_cannot_reuse_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            SampleBatchStore(output, {"seed": 1})

            with self.assertRaises(ValueError):
                SampleBatchStore(output, {"seed": 2})

    def test_nonempty_legacy_output_is_not_treated_as_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            (output / "samples.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                SampleBatchStore(output, {"seed": 1})


if __name__ == "__main__":
    unittest.main()
