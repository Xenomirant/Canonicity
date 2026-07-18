#!/usr/bin/env python3
"""Freeze and analyze the currently completed canonicity-matrix batches."""

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from transformers import AutoTokenizer

from canonicity.core import SampledSequence, evaluate_samples

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/model-matrix"
OUT = Path(__file__).resolve().parent
LENGTHS = (32, 64, 128, 256, 512, 1024, 2048)
GEMMA = "gemma3-4b-it"
QWEN = "qwen3-30b-a3b-instruct-2507"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def pooled(rows, context_ids=None):
    if context_ids is not None:
        rows = [row for row in rows if row["context_id"] in context_ids]
    output = []
    for length in LENGTHS:
        selected = [row for row in rows if int(row["length"]) == length]
        total = sum(int(row["total_samples"]) for row in selected)
        eligible = sum(int(row["eligible_sequences"]) for row in selected)
        canonical = sum(int(row["canonical_sequences"]) for row in selected)
        output.append({
            "length": length,
            "total_samples": total,
            "eligible_sequences": eligible,
            "canonical_sequences": canonical,
            "noncanonical_sequences": eligible - canonical,
            "canonical_percentage": 100 * canonical / eligible if eligible else None,
        })
    return output


def normalized(model, condition, status, scope, rows):
    output = []
    for row in rows:
        total = int(row["total_samples"])
        eligible = int(row["eligible_sequences"])
        canonical = int(row["canonical_sequences"])
        value = row["canonical_percentage"]
        output.append({
            "model": model,
            "condition": condition,
            "status": status,
            "scope": scope,
            "length": int(row["length"]),
            "total_rollouts": total,
            "eligible_rollouts": eligible,
            "survival_percentage": 100 * eligible / total if total else None,
            "canonical_rollouts": canonical,
            "noncanonical_rollouts": eligible - canonical,
            "canonical_percentage": None if value in (None, "") else float(value),
        })
    return output


def freeze_qwen():
    job = RESULTS / QWEN / "wikitext"
    manifest = json.loads((job / "sampling_plan.json").read_text())
    plan = manifest["plan"]
    # This list is the snapshot boundary. Files committed later are ignored.
    files = sorted((job / "sample_batches").glob("*.json"))
    samples = []
    positions = []
    sources = []
    for path in files:
        record = json.loads(path.read_text())
        if record["plan_fingerprint"] != manifest["plan_fingerprint"]:
            raise RuntimeError(f"plan mismatch: {path}")
        positions.append(int(record["context_position"]))
        sources.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
        samples.extend(
            SampledSequence(
                record["context_id"], int(sample["sample_index"]),
                tuple(sample["token_ids"]), sample["termination"]
            )
            for sample in record["samples"]
        )
    if positions != list(range(len(files))):
        raise RuntimeError("Qwen batches are not a contiguous prompt prefix")
    counts = Counter(sample.context_id for sample in samples)
    if set(counts.values()) != {int(plan["samples_per_context"])}:
        raise RuntimeError("Qwen snapshot includes an incomplete context")
    tokenizer = AutoTokenizer.from_pretrained(
        plan["tokenizer_id"], revision=plan["tokenizer_commit"],
        local_files_only=True
    )
    result = evaluate_samples(tokenizer, samples, LENGTHS, examples_per_length=0)
    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial ordered-prompt snapshot",
        "completed_contexts": len(files),
        "planned_contexts": len(plan["prompts"]),
        "samples_per_context": plan["samples_per_context"],
        "evaluated_rollouts": len(samples),
        "sampling_plan_fingerprint": manifest["plan_fingerprint"],
        "batch_files": sources,
    }
    return [asdict(row) for row in result.summaries], [asdict(row) for row in result.pooled_summaries], snapshot


def plot(curves, snapshot):
    styles = {
        (GEMMA, "unconditional", "all"): ("#3366cc", "-", "o", "Gemma · unconditional"),
        (GEMMA, "wikitext", "all"): ("#3366cc", "-", "s", "Gemma · WikiText (100)"),
        (QWEN, "unconditional", "all"): ("#d1495b", "--", "o", "Qwen · unconditional"),
        (QWEN, "wikitext", "partial"): ("#d1495b", "--", "s", f"Qwen · WikiText ({snapshot['completed_contexts']}/100)"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    for (model, condition, scope), (color, line, marker, label) in styles.items():
        rows = sorted((r for r in curves if r["model"] == model and r["condition"] == condition and r["scope"] == scope), key=lambda r: r["length"])
        x = [r["length"] for r in rows]
        axes[0].plot(x, [r["canonical_percentage"] for r in rows], line, color=color, marker=marker, lw=2, label=label)
        axes[1].plot(x, [r["survival_percentage"] for r in rows], line, color=color, marker=marker, lw=2, label=label)
    for axis, title, ylabel in zip(axes, ("Canonicity among survivors", "Continuation survival"), ("Canonical sequences (%)", "Rollouts reaching checkpoint (%)")):
        axis.set_xscale("log", base=2)
        axis.set_xticks(LENGTHS)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        axis.set_ylim(-2, 102)
        axis.set_xlabel("Sampled continuation tokens")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=190)
    plt.close(fig)

    gemma = {r["length"]: r for r in curves if r["model"] == GEMMA and r["scope"] == "matched"}
    qwen = {r["length"]: r for r in curves if r["model"] == QWEN and r["scope"] == "partial"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for values, color, line, label in ((gemma, "#3366cc", "-", "Gemma"), (qwen, "#d1495b", "--", "Qwen")):
        axes[0].plot(LENGTHS, [values[n]["canonical_percentage"] for n in LENGTHS], line, color=color, marker="o", lw=2.2, label=label)
    delta = [(qwen[n]["canonical_percentage"] - gemma[n]["canonical_percentage"]) if qwen[n]["canonical_percentage"] is not None and gemma[n]["canonical_percentage"] is not None else float("nan") for n in LENGTHS]
    axes[1].axhline(0, color="black", lw=1)
    axes[1].plot(LENGTHS, delta, color="#6f4e7c", marker="o", lw=2)
    for axis in axes:
        axis.set_xscale("log", base=2); axis.set_xticks(LENGTHS)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        axis.set_xlabel("Sampled continuation tokens"); axis.grid(alpha=.25)
    axes[0].set_ylim(-2, 102); axes[0].set_ylabel("Canonical sequences (%)")
    axes[0].set_title(f"Same first {snapshot['completed_contexts']} prompts"); axes[0].legend()
    axes[1].set_ylabel("Qwen − Gemma (percentage points)"); axes[1].set_title("Descriptive survivor difference")
    fig.tight_layout(); fig.savefig(OUT / "matched_wikitext.png", dpi=190); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q_context, q_pool, snapshot = freeze_qwen()
    context_ids = {row["context_id"] for row in q_context}
    g_context = read_csv(RESULTS / GEMMA / "wikitext/summary.csv")
    curves = []
    curves += normalized(GEMMA, "unconditional", "complete", "all", read_csv(RESULTS / GEMMA / "unconditional/summary.csv"))
    curves += normalized(GEMMA, "wikitext", "complete", "all", read_csv(RESULTS / GEMMA / "wikitext/pooled_summary.csv"))
    curves += normalized(QWEN, "unconditional", "complete", "all", read_csv(RESULTS / QWEN / "unconditional/summary.csv"))
    curves += normalized(QWEN, "wikitext", "partial", "partial", q_pool)
    curves += normalized(GEMMA, "wikitext", "matched subset", "matched", pooled(g_context, context_ids))
    write_csv(OUT / "canonicity_snapshot.csv", curves)
    write_csv(OUT / "qwen_wikitext_context_snapshot.csv", q_context)
    (OUT / "snapshot_manifest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    plot(curves, snapshot)
    print(json.dumps({"contexts": snapshot["completed_contexts"], "rollouts": snapshot["evaluated_rollouts"], "captured": snapshot["captured_at_utc"]}))


if __name__ == "__main__":
    main()
