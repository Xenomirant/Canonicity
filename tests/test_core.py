import unittest

from canonicity.core import (
    SampledSequence,
    compare_tokenization,
    evaluate_samples,
    extract_continuation,
)


class GreedyTokenizer:
    pieces = {1: "a", 2: "b", 3: "ab", 4: "c"}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        self.decode_options = (
            skip_special_tokens,
            clean_up_tokenization_spaces,
        )
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def encode(self, text, *, add_special_tokens):
        self.add_special_tokens = add_special_tokens
        result = []
        index = 0
        while index < len(text):
            if text.startswith("ab", index):
                result.append(3)
                index += 2
            elif text[index] == "a":
                result.append(1)
                index += 1
            elif text[index] == "b":
                result.append(2)
                index += 1
            elif text[index] == "c":
                result.append(4)
                index += 1
            else:
                raise AssertionError(f"unexpected text: {text}")
        return result


class CanonicityTests(unittest.TestCase):
    def test_exact_round_trip_defines_canonicity(self):
        tokenizer = GreedyTokenizer()

        canonical = compare_tokenization(tokenizer, [3, 4])
        noncanonical = compare_tokenization(tokenizer, [1, 2, 4])

        self.assertTrue(canonical.is_canonical)
        self.assertFalse(noncanonical.is_canonical)
        self.assertEqual(noncanonical.canonical_ids, (3, 4))
        self.assertEqual(noncanonical.first_difference, 0)
        self.assertEqual(tokenizer.decode_options, (False, False))
        self.assertFalse(tokenizer.add_special_tokens)

    def test_continuation_excludes_prompt_and_terminal_token(self):
        token_ids, termination = extract_continuation(
            [90, 91, 3, 4, 99, 1], input_width=2, terminal_ids={99}
        )

        self.assertEqual(token_ids, (3, 4))
        self.assertEqual(termination, "eos_token:99")

    def test_summary_counts_whole_prefixes_and_early_termination(self):
        tokenizer = GreedyTokenizer()
        samples = (
            SampledSequence("ctx", 0, (3, 4), "max_length"),
            SampledSequence("ctx", 1, (1, 2, 4), "max_length"),
            SampledSequence("ctx", 2, (1,), "special_token:99"),
        )

        result = evaluate_samples(
            tokenizer, samples, [1, 2, 3], examples_per_length=1
        )

        by_length = {row.length: row for row in result.summaries}
        self.assertEqual(by_length[1].eligible_sequences, 3)
        self.assertEqual(by_length[1].canonical_sequences, 3)
        self.assertEqual(by_length[2].eligible_sequences, 2)
        self.assertEqual(by_length[2].canonical_sequences, 1)
        self.assertEqual(by_length[2].canonical_percentage, 50.0)
        self.assertLess(by_length[2].canonical_ci95_low, 50.0)
        self.assertGreater(by_length[2].canonical_ci95_high, 50.0)
        self.assertEqual(by_length[3].eligible_sequences, 1)
        self.assertEqual(by_length[3].canonical_sequences, 0)
        self.assertEqual(by_length[3].canonical_ci95_low, 0.0)
        self.assertEqual(by_length[3].terminated_before_length, 2)
        self.assertEqual(len(result.examples), 2)
        self.assertEqual(
            [row.canonical_percentage for row in result.pooled_summaries],
            [row.canonical_percentage for row in result.summaries],
        )

    def test_multi_prompt_pooled_interval_is_left_descriptive(self):
        tokenizer = GreedyTokenizer()
        samples = (
            SampledSequence("one", 0, (3,), "max_length"),
            SampledSequence("two", 0, (1,), "max_length"),
        )

        result = evaluate_samples(tokenizer, samples, [1])

        self.assertEqual(result.pooled_summaries[0].canonical_percentage, 100.0)
        self.assertIsNone(result.pooled_summaries[0].canonical_ci95_low)
        self.assertIsNone(result.pooled_summaries[0].canonical_ci95_high)

    def test_evaluation_progress_advances_once_per_rollout(self):
        tokenizer = GreedyTokenizer()
        samples = (
            SampledSequence("ctx", 0, (3,), "max_length"),
            SampledSequence("ctx", 1, (4,), "max_length"),
        )
        updates = []

        evaluate_samples(
            tokenizer,
            samples,
            [1],
            progress=lambda done, total: updates.append((done, total)),
        )

        self.assertEqual(updates, [(1, 2), (2, 2)])


if __name__ == "__main__":
    unittest.main()
