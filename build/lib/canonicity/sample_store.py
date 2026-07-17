"""Transactional batch checkpoints for long model-sampling jobs."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .core import SampledSequence


SCHEMA_VERSION = 1


def plan_fingerprint(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class SampleBatchStore:
    """Validate a run plan and atomically persist each completed sample batch."""

    def __init__(self, output_dir: Path, plan: Mapping[str, Any]):
        self.output_dir = output_dir
        self.batch_dir = output_dir / "sample_batches"
        self.manifest_path = output_dir / "sampling_plan.json"
        self.plan = dict(plan)
        self.fingerprint = plan_fingerprint(self.plan)
        self._initialize()

    def _initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "plan_fingerprint": self.fingerprint,
            "plan": self.plan,
        }
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != manifest:
                raise ValueError(
                    "output directory belongs to a different sampling plan: "
                    f"{self.output_dir}"
                )
        else:
            existing_entries = [
                path
                for path in self.output_dir.iterdir()
                if path.name not in {self.batch_dir.name, self.manifest_path.name}
            ]
            if existing_entries or self.batch_dir.exists():
                raise ValueError(
                    "refusing to resume an output directory without its exact "
                    f"sampling plan: {self.output_dir}"
                )
            _atomic_json(self.manifest_path, manifest)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _batch_name(context_position: int, start: int) -> str:
        return f"context-{context_position:04d}-start-{start:06d}.json"

    def _batch_path(self, context_position: int, start: int) -> Path:
        return self.batch_dir / self._batch_name(context_position, start)

    def load_batch(
        self,
        context_position: int,
        context_id: str,
        start: int,
        expected_count: int,
    ) -> Optional[Tuple[SampledSequence, ...]]:
        path = self._batch_path(context_position, start)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("plan_fingerprint") != self.fingerprint
            or value.get("context_position") != context_position
            or value.get("context_id") != context_id
            or value.get("start") != start
        ):
            raise ValueError(f"sample batch does not match its run plan: {path}")
        records = value.get("samples")
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"sample batch has the wrong size: {path}")
        samples = tuple(
            SampledSequence(
                context_id=context_id,
                sample_index=int(record["sample_index"]),
                token_ids=tuple(int(token_id) for token_id in record["token_ids"]),
                termination=str(record["termination"]),
            )
            for record in records
        )
        if [sample.sample_index for sample in samples] != list(
            range(start, start + expected_count)
        ):
            raise ValueError(f"sample batch indices are not contiguous: {path}")
        return samples

    def write_batch(
        self,
        context_position: int,
        context_id: str,
        start: int,
        samples: Sequence[SampledSequence],
    ) -> None:
        expected_indices = list(range(start, start + len(samples)))
        if [sample.sample_index for sample in samples] != expected_indices or any(
            sample.context_id != context_id for sample in samples
        ):
            raise ValueError("cannot checkpoint a non-contiguous sample batch")
        value = {
            "schema_version": SCHEMA_VERSION,
            "plan_fingerprint": self.fingerprint,
            "context_position": context_position,
            "context_id": context_id,
            "start": start,
            "samples": [
                {
                    "sample_index": sample.sample_index,
                    "token_ids": list(sample.token_ids),
                    "termination": sample.termination,
                }
                for sample in samples
            ],
        }
        path = self._batch_path(context_position, start)
        if path.exists():
            existing = self.load_batch(
                context_position,
                context_id,
                start,
                len(samples),
            )
            if existing != tuple(samples):
                raise ValueError(f"refusing to replace a different sample batch: {path}")
            return
        _atomic_json(path, value)
