"""Model-independent definition and aggregation of tokenization canonicity."""

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence, Tuple


CANONICITY_EVALUATION_IMPLEMENTATION = "generated-text-canonicity/canonicity-v2"


class Tokenizer(Protocol):
    """The small tokenizer surface needed by the metric."""

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        ...

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        ...


@dataclass(frozen=True)
class CanonicityComparison:
    sampled_ids: Tuple[int, ...]
    canonical_ids: Tuple[int, ...]
    decoded_text: str
    is_canonical: bool
    first_difference: Optional[int]


@dataclass(frozen=True)
class SampledSequence:
    context_id: str
    sample_index: int
    token_ids: Tuple[int, ...]
    termination: str


@dataclass(frozen=True)
class PrefixStatus:
    length: int
    is_canonical: bool


@dataclass(frozen=True)
class SequenceResult:
    sample: SampledSequence
    prefixes: Tuple[PrefixStatus, ...]


@dataclass(frozen=True)
class NoncanonicalExample:
    context_id: str
    sample_index: int
    length: int
    sampled_ids: Tuple[int, ...]
    canonical_ids: Tuple[int, ...]
    decoded_text: str
    first_difference: int


@dataclass(frozen=True)
class SummaryRow:
    context_id: str
    length: int
    total_samples: int
    eligible_sequences: int
    terminated_before_length: int
    canonical_sequences: int
    noncanonical_sequences: int
    canonical_percentage: Optional[float]
    canonical_ci95_low: Optional[float]
    canonical_ci95_high: Optional[float]


@dataclass(frozen=True)
class EvaluationResult:
    summaries: Tuple[SummaryRow, ...]
    pooled_summaries: Tuple[SummaryRow, ...]
    sequences: Tuple[SequenceResult, ...]
    examples: Tuple[NoncanonicalExample, ...]


def _first_difference(left: Sequence[int], right: Sequence[int]) -> Optional[int]:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_tokenization(
    tokenizer: Tokenizer, sampled_ids: Sequence[int]
) -> CanonicityComparison:
    """Compare one sampled sequence with the canonical encoding of its text.

    The conditioning seed and terminating EOS are excluded upstream. Other
    sampled token IDs, including non-EOS control tokens, remain part of the
    measured sequence. Space cleanup is disabled because it changes the text
    whose tokenization is being measured.
    """

    sampled = tuple(int(token_id) for token_id in sampled_ids)
    decoded = tokenizer.decode(
        sampled,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    canonical = tuple(
        int(token_id)
        for token_id in tokenizer.encode(decoded, add_special_tokens=False)
    )
    difference = _first_difference(sampled, canonical)
    return CanonicityComparison(
        sampled_ids=sampled,
        canonical_ids=canonical,
        decoded_text=decoded,
        is_canonical=difference is None,
        first_difference=difference,
    )


def extract_continuation(
    output_ids: Sequence[int],
    input_width: int,
    terminal_ids: Iterable[int],
) -> Tuple[Tuple[int, ...], str]:
    """Remove conditioning input and stop before the first configured EOS."""

    if input_width < 0 or input_width > len(output_ids):
        raise ValueError("input_width must identify a prefix of output_ids")

    terminals = {int(token_id) for token_id in terminal_ids}
    continuation = []
    for token_id in output_ids[input_width:]:
        token_id = int(token_id)
        if token_id in terminals:
            return tuple(continuation), f"eos_token:{token_id}"
        continuation.append(token_id)
    return tuple(continuation), "max_length"


def _validate_lengths(lengths: Iterable[int]) -> Tuple[int, ...]:
    normalized = tuple(sorted(set(int(length) for length in lengths)))
    if not normalized or normalized[0] < 1:
        raise ValueError("lengths must contain at least one positive integer")
    return normalized


def _wilson_percentage_interval(
    successes: int, total: int
) -> Tuple[Optional[float], Optional[float]]:
    """Return a 95% Wilson score interval as percentages."""

    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    low = max(0.0, 100.0 * (centre - margin))
    high = min(100.0, 100.0 * (centre + margin))
    return low, high


def evaluate_samples(
    tokenizer: Tokenizer,
    samples: Sequence[SampledSequence],
    lengths: Iterable[int],
    *,
    examples_per_length: int = 3,
) -> EvaluationResult:
    """Compute the paper's whole-prefix canonicity rate.

    A sample is eligible at length ``n`` only if it retained at least ``n``
    continuation token IDs. Each eligible prefix contributes either one
    canonical or one non-canonical sequence; no token-level averaging is
    performed.
    """

    if examples_per_length < 0:
        raise ValueError("examples_per_length cannot be negative")
    checked_lengths = _validate_lengths(lengths)

    contexts = sorted({sample.context_id for sample in samples})
    totals = {
        context_id: sum(sample.context_id == context_id for sample in samples)
        for context_id in contexts
    }
    eligible = {
        (context_id, length): 0
        for context_id in contexts
        for length in checked_lengths
    }
    canonical = dict.fromkeys(eligible, 0)
    example_counts = dict.fromkeys(eligible, 0)

    sequence_results = []
    examples = []
    for sample in samples:
        statuses = []
        for length in checked_lengths:
            if len(sample.token_ids) < length:
                continue
            key = (sample.context_id, length)
            eligible[key] += 1
            comparison = compare_tokenization(tokenizer, sample.token_ids[:length])
            if comparison.is_canonical:
                canonical[key] += 1
            elif example_counts[key] < examples_per_length:
                assert comparison.first_difference is not None
                examples.append(
                    NoncanonicalExample(
                        context_id=sample.context_id,
                        sample_index=sample.sample_index,
                        length=length,
                        sampled_ids=comparison.sampled_ids,
                        canonical_ids=comparison.canonical_ids,
                        decoded_text=comparison.decoded_text,
                        first_difference=comparison.first_difference,
                    )
                )
                example_counts[key] += 1
            statuses.append(
                PrefixStatus(length=length, is_canonical=comparison.is_canonical)
            )
        sequence_results.append(
            SequenceResult(sample=sample, prefixes=tuple(statuses))
        )

    summaries = []
    for context_id in contexts:
        for length in checked_lengths:
            key = (context_id, length)
            denominator = eligible[key]
            canonical_count = canonical[key]
            percentage = (
                100.0 * canonical_count / denominator if denominator else None
            )
            ci_low, ci_high = _wilson_percentage_interval(
                canonical_count, denominator
            )
            summaries.append(
                SummaryRow(
                    context_id=context_id,
                    length=length,
                    total_samples=totals[context_id],
                    eligible_sequences=denominator,
                    terminated_before_length=totals[context_id] - denominator,
                    canonical_sequences=canonical_count,
                    noncanonical_sequences=denominator - canonical_count,
                    canonical_percentage=percentage,
                    canonical_ci95_low=ci_low,
                    canonical_ci95_high=ci_high,
                )
            )

    pooled_summaries = []
    for length in checked_lengths:
        length_rows = [summary for summary in summaries if summary.length == length]
        total = sum(summary.total_samples for summary in length_rows)
        denominator = sum(summary.eligible_sequences for summary in length_rows)
        canonical_count = sum(
            summary.canonical_sequences for summary in length_rows
        )
        percentage = (
            100.0 * canonical_count / denominator if denominator else None
        )
        # Rollouts that share a prompt are not an iid sample from the WikiText
        # prompt population. Keep the pooled percentage descriptive rather
        # than attaching a misleading rollout-level Wilson interval.
        ci_low, ci_high = (None, None)
        pooled_summaries.append(
            SummaryRow(
                context_id="__pooled__",
                length=length,
                total_samples=total,
                eligible_sequences=denominator,
                terminated_before_length=total - denominator,
                canonical_sequences=canonical_count,
                noncanonical_sequences=denominator - canonical_count,
                canonical_percentage=percentage,
                canonical_ci95_low=ci_low,
                canonical_ci95_high=ci_high,
            )
        )

    return EvaluationResult(
        summaries=tuple(summaries),
        pooled_summaries=tuple(pooled_summaries),
        sequences=tuple(sequence_results),
        examples=tuple(examples),
    )
