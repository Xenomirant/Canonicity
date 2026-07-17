from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import tomllib
from typing import Any


ATTACKS = (
    "canonical",
    "gcg",
    "autodan",
    "ffa",
    "advtok",
    "advtok_gcg",
    "advtok_autodan",
    "advtok_ffa",
)
DATASETS = ("advbench", "maliciousinstruct", "masterkey")
MODELS = ("llama3", "gemma2", "olmo2")


@dataclasses.dataclass(frozen=True)
class Config:
    path: pathlib.Path
    raw: dict[str, Any]
    digest: str

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]

    def model(self, key: str) -> dict[str, Any]:
        return self.raw["models"][key]


def load_config(path: str | pathlib.Path) -> Config:
    config_path = pathlib.Path(path).resolve()
    payload = config_path.read_bytes()
    raw = tomllib.loads(payload.decode("utf-8"))
    required = {"experiment", "models", "generation", "advtok", "gcg", "autodan", "strongreject"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    _validate_values(raw["experiment"].get("models", []), MODELS, "models")
    _validate_values(raw["experiment"].get("datasets", []), DATASETS, "datasets")
    _validate_values(raw["experiment"].get("attacks", []), ATTACKS, "attacks")
    return Config(config_path, raw, hashlib.sha256(payload).hexdigest())


def _validate_values(values: list[str], allowed: tuple[str, ...], label: str) -> None:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported {label}: {', '.join(invalid)}")
    if not values:
        raise ValueError(f"At least one {label[:-1]} is required")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
