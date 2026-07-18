#!/usr/bin/env python3
"""Generate a Gemma-3-4B-IT preliminary report from completed artifacts."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "results/model-matrix/gemma3-4b-it"
OUT = Path(__file__).resolve().parent
LENGTHS = (32, 64, 128, 256, 512, 1024, 2048)
COLORS = {"unconditional": "#d97706", "wikitext": "#2563eb"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows():
    output = []
    for condition, name in (("unconditional", "summary.csv"), ("wikitext", "pooled_summary.csv")):
        for row in read_csv(JOB / condition / name):
            total = int(row["total_samples"])
            eligible = int(row["eligible_sequences"])
            output.append({
                "condition": condition,
                "length": int(row["length"]),
                "total_rollouts": total,
                "eligible_rollouts": eligible,
                "survival_percentage": 100 * eligible / total,
                "canonical_rollouts": int(row["canonical_sequences"]),
                "noncanonical_rollouts": int(row["noncanonical_sequences"]),
                "canonical_percentage": "" if row["canonical_percentage"] == "" else float(row["canonical_percentage"]),
                "canonical_ci95_low": row["canonical_ci95_low"],
                "canonical_ci95_high": row["canonical_ci95_high"],
            })
    return output


def prompt_dispersion():
    rows = read_csv(JOB / "wikitext/summary.csv")
    output = []
    for length in LENGTHS:
        selected = [row for row in rows if int(row["length"]) == length]
        eligible_contexts = [row for row in selected if int(row["eligible_sequences"]) > 0]
        canonical = np.array([float(row["canonical_percentage"]) for row in eligible_contexts])
        survival = np.array([100 * int(row["eligible_sequences"]) / int(row["total_samples"]) for row in selected])
        output.append({
            "length": length,
            "contexts_with_survivors": len(eligible_contexts),
            "contexts_all_canonical": int(np.sum(canonical == 100)) if len(canonical) else 0,
            "prompt_canonicity_q25": float(np.quantile(canonical, .25)) if len(canonical) else "",
            "prompt_canonicity_median": float(np.median(canonical)) if len(canonical) else "",
            "prompt_canonicity_q75": float(np.quantile(canonical, .75)) if len(canonical) else "",
            "prompt_canonicity_min": float(np.min(canonical)) if len(canonical) else "",
            "prompt_survival_q25": float(np.quantile(survival, .25)),
            "prompt_survival_median": float(np.median(survival)),
            "prompt_survival_q75": float(np.quantile(survival, .75)),
        })
    return output


def segment_rows():
    output = []
    for condition in ("unconditional", "wikitext"):
        rows = read_csv(JOB / condition / "rollout_segments.csv")
        counts = np.array([int(row["segment_count"]) for row in rows])
        generated = np.array([int(row["generated_tokens"]) for row in rows])
        noncanonical = np.array([int(row["noncanonical_prefixes"]) for row in rows])
        longest = np.array([int(row["longest_segment"]) for row in rows])
        affected = counts > 0
        output.append({
            "condition": condition,
            "rollouts": len(rows),
            "rollouts_with_any_segment": int(np.sum(affected)),
            "rollouts_with_any_segment_percentage": 100 * float(np.mean(affected)),
            "total_segments": int(np.sum(counts)),
            "max_segments_in_one_rollout": int(np.max(counts)),
            "observed_prefix_states": int(np.sum(generated)),
            "noncanonical_prefix_states": int(np.sum(noncanonical)),
            "noncanonical_prefix_state_percentage": 100 * float(np.sum(noncanonical) / np.sum(generated)),
            "median_longest_segment_affected": float(np.median(longest[affected])) if np.any(affected) else 0,
            "max_longest_segment": int(np.max(longest)),
        })
    return output


def style_checkpoint_axis(axis, ylabel, title):
    axis.set_xscale("log", base=2)
    axis.set_xticks(LENGTHS)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axis.set_ylim(-2, 102)
    axis.set_xlabel("Sampled continuation tokens")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=.25)


def plot_overview(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    for condition, label, marker in (("unconditional", "Unconditional (64 rollouts)", "o"), ("wikitext", "WikiText (100 × 16 rollouts)", "s")):
        selected = sorted((row for row in rows if row["condition"] == condition), key=lambda row: row["length"])
        x = [row["length"] for row in selected]
        axes[0].plot(x, [np.nan if row["canonical_percentage"] == "" else row["canonical_percentage"] for row in selected], marker=marker, lw=2.3, color=COLORS[condition], label=label)
        axes[1].plot(x, [row["survival_percentage"] for row in selected], marker=marker, lw=2.3, color=COLORS[condition], label=label)
    style_checkpoint_axis(axes[0], "Canonical sequences (%)", "Canonicity among survivors")
    style_checkpoint_axis(axes[1], "Rollouts reaching checkpoint (%)", "Continuation survival")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Gemma 3 4B IT · preliminary canonicity matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "gemma_overview.png", dpi=190)
    plt.close(fig)


def plot_prompt_distributions():
    rows = read_csv(JOB / "wikitext/summary.csv")
    canon_data, survival_data = [], []
    for length in LENGTHS:
        selected = [row for row in rows if int(row["length"]) == length]
        canon_data.append([float(row["canonical_percentage"]) for row in selected if int(row["eligible_sequences"]) > 0])
        survival_data.append([100 * int(row["eligible_sequences"]) / int(row["total_samples"]) for row in selected])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for axis, values, ylabel, title in (
        (axes[0], canon_data, "Per-prompt canonicity (%)", "Canonicity across prompts with survivors"),
        (axes[1], survival_data, "Per-prompt survival (%)", "Continuation survival across all prompts"),
    ):
        axis.boxplot(values, tick_labels=[f"{x:,}" for x in LENGTHS], showfliers=True, medianprops={"color": "#dc2626", "linewidth": 1.8})
        axis.set_ylim(-2, 102)
        axis.set_xlabel("Sampled continuation tokens")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Gemma WikiText prompt heterogeneity · 16 rollouts per prompt", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "gemma_prompt_heterogeneity.png", dpi=190)
    plt.close(fig)


def plot_segments():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for condition, label in (("unconditional", "Unconditional"), ("wikitext", "WikiText")):
        rows = read_csv(JOB / condition / "rollout_segments.csv")
        affected = [row for row in rows if int(row["segment_count"]) > 0]
        axes[0].hist([int(row["first_segment_start"]) for row in affected], bins=np.geomspace(1, 2049, 18), alpha=.55, color=COLORS[condition], label=label)
        axes[1].hist([int(row["longest_segment"]) for row in affected], bins=np.geomspace(1, 2049, 18), alpha=.55, color=COLORS[condition], label=label)
    for axis, title, xlabel in ((axes[0], "First noncanonical segment", "Starting prefix position"), (axes[1], "Longest segment per affected rollout", "Segment length (prefix states)")):
        axis.set_xscale("log", base=2)
        axis.set_title(title); axis.set_xlabel(xlabel); axis.set_ylabel("Affected rollouts")
        axis.grid(alpha=.2); axis.legend()
    fig.suptitle("Gemma dense-prefix segment diagnostics", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "gemma_segments.png", dpi=190); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_rows()
    dispersion = prompt_dispersion()
    segments = segment_rows()
    write_csv(OUT / "gemma_checkpoint_summary.csv", checkpoints)
    write_csv(OUT / "gemma_prompt_dispersion.csv", dispersion)
    write_csv(OUT / "gemma_segment_summary.csv", segments)
    plot_overview(checkpoints)
    plot_prompt_distributions()
    plot_segments()
    provenance = {}
    for condition in ("unconditional", "wikitext"):
        plan = json.loads((JOB / condition / "sampling_plan.json").read_text())
        provenance[condition] = {
            "plan_fingerprint": plan["plan_fingerprint"],
            "model_commit": plan["plan"]["model_commit"],
            "tokenizer_commit": plan["plan"]["tokenizer_commit"],
            "samples_per_context": plan["plan"]["samples_per_context"],
            "context_count": len(plan["plan"]["prompts"]),
        }
    (OUT / "gemma_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoints": checkpoints, "dispersion": dispersion, "segments": segments}))


if __name__ == "__main__":
    main()
