"""Materialize long, provenance-carrying prompts from text datasets."""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _tokenize_with_offsets(tokenizer: Any, text: str) -> Tuple[List[int], List[Any]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = list(encoded["offset_mapping"])
    if len(token_ids) != len(offsets):
        raise ValueError("tokenizer returned inconsistent ids and offsets")
    return token_ids, offsets


def _token_budget_prefix(
    tokenizer: Any, text: str, prompt_tokens: int
) -> Optional[Tuple[str, int]]:
    token_ids, offsets = _tokenize_with_offsets(tokenizer, text)
    if len(token_ids) < prompt_tokens:
        return None

    # Detaching a text prefix can change the tokenizer's final merge boundary.
    # Therefore an offset from the full-text encoding is only a candidate: the
    # budget is validated against the final standalone prompt string itself.
    tried_offsets = set()
    for token_index in range(prompt_tokens - 1, -1, -1):
        end_offset = int(offsets[token_index][1])
        if end_offset <= 0 or end_offset in tried_offsets:
            continue
        tried_offsets.add(end_offset)
        prefix = text[:end_offset]
        actual_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        if actual_tokens <= prompt_tokens:
            return prefix, actual_tokens
    raise ValueError("tokenizer offsets cannot identify a non-empty prompt prefix")


def materialize_prompts(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    field: str,
    count: int,
    prompt_tokens: int,
    separator: str,
    skip_rows: int,
    provenance: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Pack sequential text rows into canonical prompt-text prefixes."""

    if count < 1 or prompt_tokens < 1:
        raise ValueError("count and prompt_tokens must be positive")
    if skip_rows < 0:
        raise ValueError("skip_rows cannot be negative")

    prompts = []
    buffer = ""
    source_start = None
    source_end = None

    for row_index, row in enumerate(rows):
        if row_index < skip_rows:
            continue
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"row {row_index} field {field!r} is not text")
        if not value:
            continue

        if buffer:
            buffer = buffer + separator + value
        else:
            buffer = value
            source_start = row_index
        source_end = row_index

        while len(prompts) < count:
            prefix = _token_budget_prefix(tokenizer, buffer, prompt_tokens)
            if prefix is None:
                break
            text, actual_tokens = prefix
            end_offset = len(text)

            prompt_number = len(prompts) + 1
            prompt_metadata = {
                **dict(provenance),
                "source_row_start": source_start,
                "source_row_end": source_end,
                "requested_prompt_tokens": prompt_tokens,
                "materializer_prompt_tokens": actual_tokens,
            }
            prompts.append(
                {
                    "id": f"dataset-{prompt_number:03d}",
                    "text": text,
                    "metadata": prompt_metadata,
                }
            )
            if len(prompts) == count:
                return tuple(prompts)

            buffer = buffer[end_offset:]
            source_start = source_end
            if not buffer:
                source_start = None
                source_end = None
                break

    raise ValueError(
        f"dataset ended after producing {len(prompts)} of {count} prompts"
    )


def materialize_document_prompts(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    field: str,
    count: int,
    prompt_tokens: int,
    separator: str,
    document_start_pattern: str,
    seed: int,
    provenance: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Select one non-overlapping prompt from each of sampled text documents."""

    if count < 1 or prompt_tokens < 1:
        raise ValueError("count and prompt_tokens must be positive")
    pattern = re.compile(document_start_pattern)
    candidates = []
    document_rows = []
    document_start = None
    document_title = None
    document_index = -1

    def finish_document(end_row: int) -> None:
        nonlocal document_rows
        finished_rows = document_rows
        document_rows = []
        if document_start is None or not finished_rows:
            return
        text = separator.join(finished_rows)
        prefix = _token_budget_prefix(tokenizer, text, prompt_tokens)
        if prefix is None:
            return
        prompt_text, actual_tokens = prefix
        candidates.append(
            {
                "text": prompt_text,
                "document_index": document_index,
                "document_title": document_title,
                "source_row_start": document_start,
                "source_row_end": end_row,
                "materializer_prompt_tokens": actual_tokens,
            }
        )

    last_row = -1
    for row_index, row in enumerate(rows):
        last_row = row_index
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"row {row_index} field {field!r} is not text")
        stripped = value.strip()
        if pattern.fullmatch(stripped):
            finish_document(row_index - 1)
            document_index += 1
            document_start = row_index
            document_title = stripped
            document_rows = [value]
        elif document_start is not None and value:
            document_rows.append(value)
    finish_document(last_row)

    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} documents meet the {prompt_tokens}-token "
            f"budget; {count} requested"
        )
    selected = random.Random(seed).sample(candidates, count)
    selected.sort(key=lambda candidate: candidate["document_index"])

    prompts = []
    for prompt_number, candidate in enumerate(selected, start=1):
        prompt_metadata = {
            **dict(provenance),
            "selection": "random documents without replacement",
            "selection_seed": seed,
            "document_start_pattern": document_start_pattern,
            "document_index": candidate["document_index"],
            "document_title": candidate["document_title"],
            "source_row_start": candidate["source_row_start"],
            "source_row_end": candidate["source_row_end"],
            "requested_prompt_tokens": prompt_tokens,
            "materializer_prompt_tokens": candidate[
                "materializer_prompt_tokens"
            ],
        }
        prompts.append(
            {
                "id": f"document-{prompt_number:03d}",
                "text": candidate["text"],
                "metadata": prompt_metadata,
            }
        )
    return tuple(prompts)


def write_prompts(path: Path, prompts: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
