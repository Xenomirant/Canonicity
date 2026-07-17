import tempfile
import unittest
from pathlib import Path

from canonicity.dataset_prompts import (
    materialize_document_prompts,
    materialize_prompts,
    write_prompts,
)
from canonicity.generation import load_prompt_file


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index, _ in enumerate(text)
            ],
        }

    def encode(self, text, *, add_special_tokens):
        return [ord(character) for character in text]


class DatasetPromptTests(unittest.TestCase):
    def test_sequential_rows_are_packed_with_provenance(self):
        prompts = materialize_prompts(
            [{"text": "ab"}, {"text": "cd"}, {"text": "efgh"}],
            CharacterTokenizer(),
            field="text",
            count=2,
            prompt_tokens=4,
            separator="",
            skip_rows=0,
            provenance={"dataset": "example/corpus"},
        )

        self.assertEqual([prompt["text"] for prompt in prompts], ["abcd", "efgh"])
        self.assertEqual(
            prompts[0]["metadata"]["materializer_prompt_tokens"], 4
        )
        self.assertEqual(prompts[0]["metadata"]["source_row_start"], 0)
        self.assertEqual(prompts[0]["metadata"]["source_row_end"], 1)

    def test_metadata_survives_jsonl_round_trip(self):
        prompts = (
            {
                "id": "dataset-001",
                "text": "abcd",
                "metadata": {"dataset": "example/corpus"},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_prompts(path, prompts)
            loaded = load_prompt_file(path)

        self.assertEqual(loaded[0].context_id, "dataset-001")
        self.assertEqual(loaded[0].metadata["dataset"], "example/corpus")

    def test_document_sampling_uses_distinct_documents(self):
        rows = [
            {"text": "= One ="},
            {"text": "abcdefgh"},
            {"text": "= Two ="},
            {"text": "ijklmnop"},
            {"text": "= Three ="},
            {"text": "qrstuvwx"},
        ]

        prompts = materialize_document_prompts(
            rows,
            CharacterTokenizer(),
            field="text",
            count=2,
            prompt_tokens=8,
            separator="",
            document_start_pattern=r"^= [^=].* =$",
            seed=3,
            provenance={"dataset": "example/corpus"},
        )

        document_indices = {
            prompt["metadata"]["document_index"] for prompt in prompts
        }
        self.assertEqual(len(document_indices), 2)
        self.assertTrue(
            all(
                prompt["metadata"]["selection"]
                == "random documents without replacement"
                for prompt in prompts
            )
        )


if __name__ == "__main__":
    unittest.main()
