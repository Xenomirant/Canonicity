import math
import unittest

from canonicity.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    benjamini_yekutieli_log,
)


class MultipleTestingTests(unittest.TestCase):
    def test_bh_preserves_order_and_monotonicity(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])

        self.assertAlmostEqual(adjusted[0], 0.02)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertAlmostEqual(adjusted[3], 0.008)

    def test_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.01, math.nan])

    def test_by_is_conservative_under_arbitrary_dependence(self):
        adjusted = benjamini_yekutieli([0.01, 0.2])

        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.3)

    def test_log_by_matches_ordinary_values_without_underflow(self):
        p_values = [0.01, 0.2]

        log_adjusted = benjamini_yekutieli_log(
            [math.log(value) for value in p_values]
        )

        self.assertEqual(
            tuple(round(math.exp(value), 14) for value in log_adjusted),
            tuple(round(value, 14) for value in benjamini_yekutieli(p_values)),
        )

    def test_log_by_preserves_extreme_tail_information(self):
        log_adjusted = benjamini_yekutieli_log([-1000.0, -900.0, 0.0])

        self.assertLess(log_adjusted[0], log_adjusted[1])
        self.assertTrue(all(math.isfinite(value) for value in log_adjusted))
        self.assertEqual(math.exp(log_adjusted[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
