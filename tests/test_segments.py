import math
import unittest

from canonicity.segments import (
    NoncanonicalSegment,
    RolloutSegmentResult,
    _exact_stratified_two_sided,
    _exact_stratified_two_sided_log,
    recurrence_rows,
    segments_from_noncanonical_flags,
)


def rollout(context, index, tokens, segments):
    return RolloutSegmentResult(
        context_id=context,
        sample_index=index,
        generated_tokens=tokens,
        termination="max_length",
        noncanonical_prefixes=sum(segment.length for segment in segments),
        segment_count=len(segments),
        first_segment_start=segments[0].start if segments else None,
        last_segment_end=segments[-1].end if segments else None,
        longest_segment=max((segment.length for segment in segments), default=0),
        segments=tuple(segments),
    )


class SegmentTests(unittest.TestCase):
    def test_maximal_runs_are_segments(self):
        segments = segments_from_noncanonical_flags(
            [False, True, True, False, True, False]
        )

        self.assertEqual(
            segments,
            (NoncanonicalSegment(2, 3), NoncanonicalSegment(5, 5)),
        )

    def test_crossing_midpoint_is_not_false_recurrence(self):
        results = (
            rollout("ctx", 0, 8, [NoncanonicalSegment(2, 6)]),
            rollout(
                "ctx",
                1,
                8,
                [NoncanonicalSegment(2, 3), NoncanonicalSegment(7, 7)],
            ),
            rollout("ctx", 2, 8, [NoncanonicalSegment(7, 7)]),
            rollout("ctx", 3, 8, []),
            rollout("ctx", 4, 8, [NoncanonicalSegment(2, 3)]),
        )

        row = recurrence_rows(results, [8])[0]

        self.assertEqual(row.surviving_rollouts, 5)
        self.assertEqual(row.excluded_noncanonical_at_landmark, 1)
        self.assertEqual(row.at_risk_rollouts, 4)
        self.assertEqual(row.prior_and_future, 1)
        self.assertEqual(row.prior_only, 1)
        self.assertEqual(row.future_only, 1)
        self.assertEqual(row.neither, 1)
        self.assertEqual(row.pooled_future_probability_given_prior, 0.5)
        self.assertEqual(row.pooled_future_probability_given_no_prior, 0.5)
        self.assertEqual(row.test_name, "fisher_exact")
        self.assertEqual(row.p_value, 1.0)
        self.assertEqual(row.common_odds_ratio, 1.0)

    def test_early_termination_is_excluded_at_horizon(self):
        results = (
            rollout("ctx", 0, 8, []),
            rollout("ctx", 1, 7, [NoncanonicalSegment(2, 2)]),
        )

        row = recurrence_rows(results, [8])[0]

        self.assertEqual(row.surviving_rollouts, 1)
        self.assertEqual(row.at_risk_rollouts, 1)

    def test_sparse_stratified_test_reduces_to_exact_fisher(self):
        results = (
            # The first prompt's at-risk table is [2, 0; 1, 1].
            rollout("one", 0, 8, [NoncanonicalSegment(2, 2), NoncanonicalSegment(7, 7)]),
            rollout("one", 1, 8, [NoncanonicalSegment(2, 2), NoncanonicalSegment(7, 7)]),
            rollout("one", 2, 8, [NoncanonicalSegment(7, 7)]),
            rollout("one", 3, 8, []),
            # The second prompt is degenerate and adds no information.
            rollout("two", 0, 8, []),
            rollout("two", 1, 8, []),
        )

        row = recurrence_rows(results, [8])[0]

        self.assertEqual(row.informative_contexts, 1)
        self.assertEqual(row.test_name, "exact_stratified_conditional")
        self.assertEqual(row.p_value, 1.0)
        self.assertEqual(row.log10_p_value, 0.0)

    def test_exact_stratified_probability_ordering_reference(self):
        tables = ((3, 1, 1, 3), (2, 2, 1, 3))

        p_value = _exact_stratified_two_sided(tables)

        self.assertIsNotNone(p_value)
        self.assertAlmostEqual(p_value, 23 / 70, places=14)

    def test_exact_stratified_extreme_tails_do_not_hit_a_numeric_floor(self):
        moderate = [(24, 8, 8, 24)] * 5
        stronger = [(32, 0, 0, 32)] * 5

        moderate_log_p = _exact_stratified_two_sided_log(moderate)
        stronger_log_p = _exact_stratified_two_sided_log(stronger)

        self.assertIsNotNone(moderate_log_p)
        self.assertIsNotNone(stronger_log_p)
        self.assertLess(stronger_log_p, moderate_log_p)
        self.assertLess(
            _exact_stratified_two_sided(stronger),
            _exact_stratified_two_sided(moderate),
        )

    def test_log_p_value_remains_finite_below_float_probability_range(self):
        log_p_value = _exact_stratified_two_sided_log(
            [(32, 0, 0, 32)] * 100
        )

        self.assertIsNotNone(log_p_value)
        self.assertTrue(log_p_value < math.log(float.fromhex("0x0.0000000000001p-1022")))
        self.assertEqual(
            _exact_stratified_two_sided([(32, 0, 0, 32)] * 100),
            0.0,
        )

    def test_horizons_must_be_even(self):
        with self.assertRaises(ValueError):
            recurrence_rows((), [7])


if __name__ == "__main__":
    unittest.main()
