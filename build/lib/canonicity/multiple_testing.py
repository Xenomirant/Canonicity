"""Multiple-testing correction for planned recurrence-test families."""

import math
from typing import Iterable, Tuple


def benjamini_hochberg(p_values: Iterable[float]) -> Tuple[float, ...]:
    """Return Benjamini-Hochberg adjusted p-values in original order."""

    values = tuple(float(value) for value in p_values)
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in values
    ):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    count = len(values)
    if count == 0:
        return ()

    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running_minimum = 1.0
    for reverse_rank, (original_index, value) in enumerate(
        reversed(ordered), start=1
    ):
        rank = count - reverse_rank + 1
        running_minimum = min(running_minimum, value * count / rank)
        adjusted[original_index] = min(1.0, running_minimum)
    return tuple(adjusted)


def benjamini_yekutieli(p_values: Iterable[float]) -> Tuple[float, ...]:
    """Return FDR-adjusted values valid under arbitrary test dependence."""

    values = tuple(float(value) for value in p_values)
    if not values:
        return ()
    harmonic = sum(1.0 / rank for rank in range(1, len(values) + 1))
    return tuple(
        min(1.0, adjusted * harmonic)
        for adjusted in benjamini_hochberg(values)
    )


def benjamini_yekutieli_log(
    log_p_values: Iterable[float],
) -> Tuple[float, ...]:
    """Return natural-log BY adjusted values without tail underflow."""

    values = tuple(float(value) for value in log_p_values)
    if any(not math.isfinite(value) or value > 0.0 for value in values):
        raise ValueError("log p-values must be finite and no greater than zero")
    count = len(values)
    if count == 0:
        return ()

    log_multiplier = math.log(count) + math.log(
        sum(1.0 / rank for rank in range(1, count + 1))
    )
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_minimum = 0.0
    for reverse_rank, (original_index, log_p_value) in enumerate(
        reversed(ordered), start=1
    ):
        rank = count - reverse_rank + 1
        candidate = log_p_value + log_multiplier - math.log(rank)
        running_minimum = min(running_minimum, candidate)
        adjusted[original_index] = running_minimum
    return tuple(adjusted)
