"""Dense prefix-state segments and recurrence association statistics."""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .core import SampledSequence, Tokenizer, compare_tokenization


SEGMENT_ANALYSIS_IMPLEMENTATION = "generated-text-canonicity/segments-v2"


@dataclass(frozen=True)
class NoncanonicalSegment:
    """A maximal run of non-canonical generated-prefix states, inclusive."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class RolloutSegmentResult:
    context_id: str
    sample_index: int
    generated_tokens: int
    termination: str
    noncanonical_prefixes: int
    segment_count: int
    first_segment_start: Optional[int]
    last_segment_end: Optional[int]
    longest_segment: int
    segments: Tuple[NoncanonicalSegment, ...]


@dataclass(frozen=True)
class RecurrenceRow:
    horizon: int
    landmark: int
    surviving_rollouts: int
    at_risk_rollouts: int
    excluded_noncanonical_at_landmark: int
    eligible_contexts: int
    informative_contexts: int
    prior_and_future: int
    prior_only: int
    future_only: int
    neither: int
    pooled_future_probability_given_prior: Optional[float]
    pooled_future_probability_given_no_prior: Optional[float]
    pooled_future_risk_difference: Optional[float]
    common_odds_ratio: Optional[float]
    test_name: Optional[str]
    test_statistic: Optional[float]
    p_value: Optional[float]
    log10_p_value: Optional[float]


@dataclass(frozen=True)
class SegmentAnalysis:
    rollouts: Tuple[RolloutSegmentResult, ...]
    recurrence: Tuple[RecurrenceRow, ...]


def segments_from_noncanonical_flags(
    flags: Iterable[bool],
) -> Tuple[NoncanonicalSegment, ...]:
    """Convert one Boolean state per prefix into maximal true runs."""

    segments = []
    start = None
    final_position = 0
    for position, is_noncanonical in enumerate(flags, start=1):
        final_position = position
        if is_noncanonical and start is None:
            start = position
        elif not is_noncanonical and start is not None:
            segments.append(NoncanonicalSegment(start=start, end=position - 1))
            start = None
    if start is not None:
        segments.append(NoncanonicalSegment(start=start, end=final_position))
    return tuple(segments)


def _rollout_segments(
    tokenizer: Tokenizer, sample: SampledSequence
) -> RolloutSegmentResult:
    flags = []
    for length in range(1, len(sample.token_ids) + 1):
        comparison = compare_tokenization(tokenizer, sample.token_ids[:length])
        flags.append(not comparison.is_canonical)

    segments = segments_from_noncanonical_flags(flags)
    return RolloutSegmentResult(
        context_id=sample.context_id,
        sample_index=sample.sample_index,
        generated_tokens=len(sample.token_ids),
        termination=sample.termination,
        noncanonical_prefixes=sum(flags),
        segment_count=len(segments),
        first_segment_start=segments[0].start if segments else None,
        last_segment_end=segments[-1].end if segments else None,
        longest_segment=max((segment.length for segment in segments), default=0),
        segments=segments,
    )


def _validate_horizons(horizons: Iterable[int]) -> Tuple[int, ...]:
    checked = tuple(sorted(set(int(horizon) for horizon in horizons)))
    if not checked or checked[0] < 2 or any(horizon % 2 for horizon in checked):
        raise ValueError("recurrence horizons must be positive even integers >= 2")
    return checked


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _odds_ratio(a: int, b: int, c: int, d: int) -> Optional[float]:
    numerator = a * d
    denominator = b * c
    if denominator == 0:
        return math.inf if numerator > 0 else None
    return numerator / denominator


def _mantel_haenszel_summary(
    tables: Sequence[Tuple[int, int, int, int]],
) -> Tuple[int, Optional[float]]:
    """Return the number of informative strata and common odds ratio."""

    odds_numerator = 0.0
    odds_denominator = 0.0
    informative = 0
    for a, b, c, d in tables:
        total = a + b + c + d
        if total == 0:
            continue
        odds_numerator += a * d / total
        odds_denominator += b * c / total
        if total <= 1:
            continue
        row_one = a + b
        row_zero = c + d
        column_one = a + c
        column_zero = b + d
        stratum_variance = (
            row_one
            * row_zero
            * column_one
            * column_zero
            / (total * total * (total - 1))
        )
        if stratum_variance == 0:
            continue
        informative += 1

    common_or = (
        odds_numerator / odds_denominator
        if odds_denominator > 0
        else (math.inf if odds_numerator > 0 else None)
    )
    return informative, common_or


def _logaddexp(left: float, right: float) -> float:
    """Add two log probabilities without leaving log space."""

    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def _logsumexp(values: Iterable[float]) -> float:
    total = float("-inf")
    for value in values:
        total = _logaddexp(total, value)
    return total


def _hypergeometric_log_distribution(
    table: Tuple[int, int, int, int],
) -> Dict[int, float]:
    """Log conditional null distribution of the upper-left cell."""

    a, b, c, d = table
    row_one = a + b
    row_zero = c + d
    column_one = a + c
    total = row_one + row_zero
    low = max(0, column_one - row_zero)
    high = min(row_one, column_one)
    log_probabilities: Dict[int, float] = {
        value: (
            _log_comb(row_one, value)
            + _log_comb(row_zero, column_one - value)
            - _log_comb(total, column_one)
        )
        for value in range(low, high + 1)
    }
    log_normalizer = _logsumexp(log_probabilities.values())
    return {
        value: log_probability - log_normalizer
        for value, log_probability in log_probabilities.items()
    }


def _exact_stratified_two_sided_log(
    tables: Sequence[Tuple[int, int, int, int]],
) -> Optional[float]:
    """Natural log exact conditional p-value for prompt-stratified tables.

    Conditioning on every stratum's margins makes the upper-left cells
    independent hypergeometric variables under the null. Their sum's null
    distribution is obtained by convolution. Probability ordering matches the
    two-sided Fisher definition and reduces to Fisher for one stratum.
    """

    informative_tables = []
    observed_sum = 0
    for table in tables:
        distribution = _hypergeometric_log_distribution(table)
        if len(distribution) > 1:
            informative_tables.append(distribution)
            observed_sum += table[0]
    if not informative_tables:
        return None

    total_distribution: Dict[int, float] = {0: 0.0}
    for stratum_distribution in informative_tables:
        convolved: Dict[int, float] = {}
        for running_total, running_log_probability in total_distribution.items():
            for value, log_probability in stratum_distribution.items():
                combined = running_total + value
                convolved[combined] = _logaddexp(
                    convolved.get(combined, float("-inf")),
                    running_log_probability + log_probability,
                )
        log_normalizer = _logsumexp(convolved.values())
        total_distribution = {
            value: log_probability - log_normalizer
            for value, log_probability in convolved.items()
        }

    observed_log_probability = total_distribution[observed_sum]
    # The tolerance is in log-probability space. It only reconciles numerical
    # roundoff between theoretically tied probabilities; unlike an absolute
    # probability tolerance it cannot impose a floor on extreme p-values.
    included = (
        log_probability
        for log_probability in total_distribution.values()
        if log_probability <= observed_log_probability + 1e-12
    )
    return min(0.0, _logsumexp(included))


def _exact_stratified_two_sided(
    tables: Sequence[Tuple[int, int, int, int]],
) -> Optional[float]:
    """Exact conditional two-sided p-value for prompt-stratified tables."""

    log_p_value = _exact_stratified_two_sided_log(tables)
    return math.exp(log_p_value) if log_p_value is not None else None


def recurrence_rows(
    rollouts: Sequence[RolloutSegmentResult], horizons: Iterable[int]
) -> Tuple[RecurrenceRow, ...]:
    """Run a fixed-landmark recurrence association test.

    Rollouts must reach the horizon and be canonical at its halfway landmark,
    making every included rollout at risk for a new segment. Recovered rollouts
    with a completed prior segment are compared with event-free rollouts.
    Prompted runs are stratified by context.
    """

    checked_horizons = _validate_horizons(horizons)
    rows = []
    for horizon in checked_horizons:
        landmark = horizon // 2
        tables_by_context: Dict[str, list[int]] = {}
        surviving = 0
        excluded_at_landmark = 0
        for rollout in rollouts:
            if rollout.generated_tokens < horizon:
                continue
            surviving += 1
            if any(
                segment.start <= landmark <= segment.end
                for segment in rollout.segments
            ):
                excluded_at_landmark += 1
                continue
            prior = any(segment.end < landmark for segment in rollout.segments)
            future = any(
                landmark < segment.start <= horizon
                for segment in rollout.segments
            )
            table = tables_by_context.setdefault(rollout.context_id, [0, 0, 0, 0])
            if prior and future:
                table[0] += 1
            elif prior:
                table[1] += 1
            elif future:
                table[2] += 1
            else:
                table[3] += 1

        tables = [tuple(table) for table in tables_by_context.values()]
        a = sum(table[0] for table in tables)
        b = sum(table[1] for table in tables)
        c = sum(table[2] for table in tables)
        d = sum(table[3] for table in tables)
        prior_total = a + b
        no_prior_total = c + d
        prior_probability = a / prior_total if prior_total else None
        no_prior_probability = c / no_prior_total if no_prior_total else None
        risk_difference = (
            prior_probability - no_prior_probability
            if prior_probability is not None and no_prior_probability is not None
            else None
        )

        if len(tables) == 1:
            informative = int(
                (a + b) > 0
                and (c + d) > 0
                and (a + c) > 0
                and (b + d) > 0
            )
            test_name = "fisher_exact" if informative else None
            statistic = None
            log_p_value = (
                _exact_stratified_two_sided_log(tables) if informative else None
            )
            p_value = math.exp(log_p_value) if log_p_value is not None else None
            log10_p_value = (
                log_p_value / math.log(10.0)
                if log_p_value is not None
                else None
            )
            common_or = _odds_ratio(a, b, c, d)
        else:
            informative, common_or = _mantel_haenszel_summary(tables)
            statistic = None
            log_p_value = _exact_stratified_two_sided_log(tables)
            p_value = math.exp(log_p_value) if log_p_value is not None else None
            log10_p_value = (
                log_p_value / math.log(10.0)
                if log_p_value is not None
                else None
            )
            test_name = "exact_stratified_conditional" if informative else None

        rows.append(
            RecurrenceRow(
                horizon=horizon,
                landmark=landmark,
                surviving_rollouts=surviving,
                at_risk_rollouts=a + b + c + d,
                excluded_noncanonical_at_landmark=excluded_at_landmark,
                eligible_contexts=len(tables),
                informative_contexts=informative,
                prior_and_future=a,
                prior_only=b,
                future_only=c,
                neither=d,
                pooled_future_probability_given_prior=prior_probability,
                pooled_future_probability_given_no_prior=no_prior_probability,
                pooled_future_risk_difference=risk_difference,
                common_odds_ratio=common_or,
                test_name=test_name,
                test_statistic=statistic,
                p_value=p_value,
                log10_p_value=log10_p_value,
            )
        )
    return tuple(rows)


def analyze_segments(
    tokenizer: Tokenizer,
    samples: Sequence[SampledSequence],
    horizons: Iterable[int],
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    workers: int = 1,
) -> SegmentAnalysis:
    """Evaluate every generated prefix and summarize non-canonical runs."""

    if workers < 1:
        raise ValueError("segment workers must be positive")
    checked_horizons = _validate_horizons(horizons)
    rollouts = []
    total = len(samples)
    evaluate = partial(_rollout_segments, tokenizer)
    if workers == 1:
        evaluated = map(evaluate, samples)
        for index, rollout in enumerate(evaluated, start=1):
            rollouts.append(rollout)
            if progress is not None:
                progress(index, total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, rollout in enumerate(
                executor.map(evaluate, samples), start=1
            ):
                rollouts.append(rollout)
                if progress is not None:
                    progress(index, total)
    return SegmentAnalysis(
        rollouts=tuple(rollouts),
        recurrence=recurrence_rows(rollouts, checked_horizons),
    )
