from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import re
import urllib.request
from dataclasses import asdict, dataclass
from typing import Iterable


SOURCES = {
    "advbench": {
        "url": "https://raw.githubusercontent.com/SheltonLiu-N/AutoDAN/34062e964185693e81a6775b4f0d00bfd7507612/data/advbench/harmful_behaviors.csv",
        "sha256": "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
        "count": 520,
    },
    "maliciousinstruct": {
        "url": "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/a1e490531c0ef4bc13430178bf1fbc8f813e34b1a/data/MaliciousInstruct.txt",
        "sha256": "a64d69dd1004aac4a23aaff3748fb8f17161893f74db36f45b8bc849eb1bcce5",
        "count": 100,
    },
    "masterkey": {
        "url": "https://docs.google.com/spreadsheets/d/1mA7g28YLAhCPSNT1YLW-F0A0C1qzzs4SQZTmWjj4DvU/export?format=csv",
        "sha256": "cd656a470b22c5b1f09ab197bfda7fd8f76a8ff49a91e2172f72bc7094330e62",
        "count": 45,
    },
}


@dataclass(frozen=True)
class Prompt:
    id: str
    dataset: str
    prompt: str
    target: str
    target_origin: str
    category: str | None
    source: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "advtok-replication/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def affirmative_prefix(prompt: str) -> str:
    """Derive a reproducible optimization target when the source has none.

    The paper says response prefixes were manually constructed but does not publish
    them. This deterministic rule is consequently an explicit approximation.
    """
    text = prompt.strip().rstrip("?.!")
    patterns = (
        (r"(?i)^how to\s+", "Sure, here is how to "),
        (r"(?i)^how can (?:i|we|you)\s+", "Sure, here is how to "),
        (r"(?i)^what are some\s+", "Sure, here are some "),
        (r"(?i)^what is the\s+", "Sure, the "),
        (r"(?i)^ways to\s+", "Sure, here are ways to "),
        (r"(?i)^tips for\s+", "Sure, here are tips for "),
        (r"(?i)^ideas for\s+", "Sure, here are ideas for "),
    )
    for pattern, replacement in patterns:
        if re.search(pattern, text):
            return re.sub(pattern, replacement, text, count=1)
    return f"Sure, here is a detailed answer to the request: {text}"


def parse_advbench(payload: bytes, source: str) -> list[Prompt]:
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    return [
        Prompt(
            id=f"advbench-{index:04d}",
            dataset="advbench",
            prompt=row["goal"].strip(),
            target=row["target"].strip(),
            target_origin="source",
            category=None,
            source=source,
        )
        for index, row in enumerate(rows)
        if row.get("goal", "").strip()
    ]


def parse_maliciousinstruct(payload: bytes, source: str) -> list[Prompt]:
    prompts = [line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    return [
        Prompt(
            id=f"maliciousinstruct-{index:04d}",
            dataset="maliciousinstruct",
            prompt=prompt,
            target=affirmative_prefix(prompt),
            target_origin="derived-unreleased-paper-prefix",
            category=None,
            source=source,
        )
        for index, prompt in enumerate(prompts)
    ]


def parse_masterkey(payload: bytes, source: str) -> list[Prompt]:
    rows = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    current_category: str | None = None
    parsed: list[Prompt] = []
    for row in rows:
        if not row or (row[0].strip().lower() == "questions"):
            continue
        if row[0].strip():
            current_category = row[0].strip()
        prompt = row[1].strip() if len(row) > 1 else ""
        if not prompt:
            continue
        parsed.append(
            Prompt(
                id=f"masterkey-{len(parsed):04d}",
                dataset="masterkey",
                prompt=prompt,
                target=affirmative_prefix(prompt),
                target_origin="derived-unreleased-paper-prefix",
                category=current_category,
                source=source,
            )
        )
    return parsed


PARSERS = {
    "advbench": parse_advbench,
    "maliciousinstruct": parse_maliciousinstruct,
    "masterkey": parse_masterkey,
}


def write_jsonl(path: pathlib.Path, rows: Iterable[Prompt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def fetch_all(root: pathlib.Path, force: bool = False) -> dict[str, dict[str, object]]:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for name, metadata in SOURCES.items():
        raw_path = raw_dir / f"{name}.source"
        if raw_path.exists() and not force:
            payload = raw_path.read_bytes()
        else:
            payload = download(str(metadata["url"]))
            raw_path.write_bytes(payload)
        actual_hash = sha256(payload)
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(
                f"{name}: SHA-256 mismatch; expected {metadata['sha256']}, got {actual_hash}. "
                "The public source may have changed."
            )
        prompts = PARSERS[name](payload, str(metadata["url"]))
        if len(prompts) != metadata["count"]:
            raise RuntimeError(f"{name}: expected {metadata['count']} prompts, parsed {len(prompts)}")
        write_jsonl(processed_dir / f"{name}.jsonl", prompts)
        manifest[name] = {
            **metadata,
            "raw_file": str(raw_path.relative_to(root)),
            "processed_file": str((processed_dir / f"{name}.jsonl").relative_to(root)),
            "target_note": "source" if name == "advbench" else "deterministically derived; paper prefixes unavailable",
        }
    manifest_path = root / "data" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_prompts(root: pathlib.Path, dataset: str, limit: int = 0) -> list[dict[str, object]]:
    path = root / "data" / "processed" / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/fetch_data.sh first")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify the three paper datasets")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = fetch_all(args.root.resolve(), force=args.force)
    for name, metadata in manifest.items():
        print(f"{name}: {metadata['count']} rows, sha256={metadata['sha256']}")


if __name__ == "__main__":
    main()
