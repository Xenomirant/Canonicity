#!/usr/bin/env python3
"""Generate final cross-model summaries and figures from validated matrix data."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/model-matrix"
OUT = Path(__file__).resolve().parent
LENGTHS = (32, 64, 128, 256, 512, 1024, 2048)
HORIZONS = (128, 256, 512, 1024, 2048)
MODELS = (
    "gemma3-4b-it",
    "qwen3-30b-a3b-instruct-2507",
    "gemma-2b-it",
    "llama2-7b",
    "mamba-130m",
)
LABELS = {
    "gemma3-4b-it": "Gemma 3 4B IT",
    "qwen3-30b-a3b-instruct-2507": "Qwen3 30B-A3B",
    "gemma-2b-it": "Gemma 1 2B IT",
    "llama2-7b": "Llama 2 7B",
    "mamba-130m": "Mamba 130M",
}
COLORS = {
    "gemma3-4b-it": "#2563eb",
    "qwen3-30b-a3b-instruct-2507": "#dc2626",
    "gemma-2b-it": "#7c3aed",
    "llama2-7b": "#059669",
    "mamba-130m": "#d97706",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(raw: str | None) -> float | None:
    return None if raw in (None, "") else float(raw)


def checkpoint_summary() -> list[dict]:
    output = []
    for raw in read_csv(RESULTS / "canonicity_all.csv"):
        total = int(raw["total_samples"])
        eligible = int(raw["eligible_sequences"])
        canonical = int(raw["canonical_sequences"])
        output.append({
            **raw,
            "model_label": LABELS[raw["model_alias"]],
            "survival_percentage": 100 * eligible / total,
            "canonical_yield_percentage": 100 * canonical / total,
        })
    return output


def segment_summary() -> list[dict]:
    output = []
    for model in MODELS:
        for condition in ("unconditional", "wikitext"):
            rows = read_csv(RESULTS / model / condition / "rollout_segments.csv")
            counts = np.array([int(row["segment_count"]) for row in rows])
            generated = np.array([int(row["generated_tokens"]) for row in rows])
            states = np.array([int(row["noncanonical_prefixes"]) for row in rows])
            longest = np.array([int(row["longest_segment"]) for row in rows])
            affected = counts > 0
            output.append({
                "model_alias": model,
                "model_label": LABELS[model],
                "condition": condition,
                "rollouts": len(rows),
                "rollouts_with_segment": int(np.sum(affected)),
                "rollouts_with_segment_percentage": 100 * float(np.mean(affected)),
                "total_segments": int(np.sum(counts)),
                "median_segments_affected": float(np.median(counts[affected])) if np.any(affected) else 0,
                "maximum_segments": int(np.max(counts)),
                "observed_prefix_states": int(np.sum(generated)),
                "noncanonical_prefix_states": int(np.sum(states)),
                "noncanonical_prefix_state_percentage": 100 * float(np.sum(states) / np.sum(generated)),
                "median_longest_segment_affected": float(np.median(longest[affected])) if np.any(affected) else 0,
                "maximum_segment_length": int(np.max(longest)),
            })
    return output


def prompt_survival_summary() -> list[dict]:
    output = []
    for model in MODELS:
        rows = read_csv(RESULTS / model / "wikitext/summary.csv")
        for length in LENGTHS:
            selected = [row for row in rows if int(row["length"]) == length]
            eligible = np.array([int(row["eligible_sequences"]) for row in selected])
            output.append({
                "model_alias": model,
                "model_label": LABELS[model],
                "length": length,
                "prompts_with_survivors": int(np.sum(eligible > 0)),
                "median_survivors_per_prompt": float(np.median(eligible)),
                "q25_survivors_per_prompt": float(np.quantile(eligible, .25)),
                "q75_survivors_per_prompt": float(np.quantile(eligible, .75)),
            })
    return output


def plot_overview(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), sharex=True, sharey="col")
    for row_index, condition in enumerate(("unconditional", "wikitext")):
        for model in MODELS:
            selected = sorted(
                (row for row in rows if row["model_alias"] == model and row["condition"] == condition),
                key=lambda row: int(row["length"]),
            )
            x = [int(row["length"]) for row in selected]
            axes[row_index, 0].plot(x, [np.nan if row["canonical_percentage"] == "" else float(row["canonical_percentage"]) for row in selected], marker="o", lw=2, color=COLORS[model], label=LABELS[model])
            axes[row_index, 1].plot(x, [float(row["survival_percentage"]) for row in selected], marker="o", lw=2, color=COLORS[model], label=LABELS[model])
        axes[row_index, 0].set_ylabel(f"{condition.title()}\nPercent")
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(LENGTHS)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        axis.set_ylim(-2, 102)
        axis.grid(alpha=.25)
    axes[0, 0].set_title("Canonicity among survivors")
    axes[0, 1].set_title("Continuation survival")
    axes[1, 0].set_xlabel("Sampled continuation tokens")
    axes[1, 1].set_xlabel("Sampled continuation tokens")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Validated five-model canonicity matrix", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "matrix_overview.png", dpi=190)
    plt.close(fig)


def plot_yield(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    for axis, condition in zip(axes, ("unconditional", "wikitext")):
        for model in MODELS:
            selected = sorted(
                (row for row in rows if row["model_alias"] == model and row["condition"] == condition),
                key=lambda row: int(row["length"]),
            )
            axis.plot([int(row["length"]) for row in selected], [float(row["canonical_yield_percentage"]) for row in selected], marker="o", lw=2, color=COLORS[model], label=LABELS[model])
        axis.set_xscale("log", base=2)
        axis.set_xticks(LENGTHS)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        axis.set_ylim(-2, 102)
        axis.set_xlabel("Sampled continuation tokens")
        axis.set_title(condition.title())
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Canonical and reached checkpoint (% of starts)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Canonical yield from all initiated rollouts", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "canonical_yield.png", dpi=190)
    plt.close(fig)


def plot_segments(rows: list[dict]) -> None:
    labels = [LABELS[model] for model in MODELS]
    x = np.arange(len(MODELS))
    width = .36
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for offset, condition in ((-width / 2, "unconditional"), (width / 2, "wikitext")):
        selected = {row["model_alias"]: row for row in rows if row["condition"] == condition}
        axes[0].bar(x + offset, [selected[model]["rollouts_with_segment_percentage"] for model in MODELS], width, color="#d97706" if condition == "unconditional" else "#2563eb", label=condition.title())
        axes[1].bar(x + offset, [selected[model]["noncanonical_prefix_state_percentage"] for model in MODELS], width, color="#d97706" if condition == "unconditional" else "#2563eb", label=condition.title())
    for axis, title, ylabel in ((axes[0], "Rollouts ever entering a segment", "Rollouts (%)"), (axes[1], "Exposure-weighted noncanonical states", "Observed prefix states (%)")):
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
    axes[0].legend()
    fig.suptitle("Dense-prefix noncanonical segment diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "segment_summary.png", dpi=190)
    plt.close(fig)


def plot_recurrence(rows: list[dict]) -> None:
    row_keys = [(model, condition) for model in MODELS for condition in ("unconditional", "wikitext")]
    risk = np.full((len(row_keys), len(HORIZONS)), np.nan)
    evidence = np.full_like(risk, np.nan)
    q_values = np.full_like(risk, np.nan)
    lookup = {(row["model_alias"], row["condition"], int(row["horizon"])): row for row in rows}
    for i, (model, condition) in enumerate(row_keys):
        for j, horizon in enumerate(HORIZONS):
            row = lookup[(model, condition, horizon)]
            if row["p_value"] != "":
                risk[i, j] = float(row["pooled_future_risk_difference"])
                q_values[i, j] = float(row["by_q_value"])
                evidence[i, j] = min(16, -math.log10(max(q_values[i, j], 1e-300)))
    labels = [f"{LABELS[model]} · {'U' if condition == 'unconditional' else 'W'}" for model, condition in row_keys]
    cmap_risk = plt.get_cmap("RdBu_r").copy(); cmap_risk.set_bad("#e5e7eb")
    cmap_q = plt.get_cmap("viridis").copy(); cmap_q.set_bad("#e5e7eb")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.1))
    images = [
        axes[0].imshow(risk, aspect="auto", cmap=cmap_risk, vmin=-1, vmax=1),
        axes[1].imshow(evidence, aspect="auto", cmap=cmap_q, vmin=0, vmax=16),
    ]
    for axis in axes:
        axis.set_xticks(range(len(HORIZONS)), [f"{value:,}" for value in HORIZONS])
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Recurrence horizon")
    axes[0].set_title("Later-risk difference: prior segment − none")
    axes[1].set_title("BY-adjusted evidence: −log10(q)")
    for i in range(len(row_keys)):
        for j in range(len(HORIZONS)):
            if not np.isnan(risk[i, j]):
                axes[0].text(j, i, f"{risk[i, j]:+.2f}", ha="center", va="center", fontsize=8)
                marker = "★" if q_values[i, j] < .05 else ""
                axes[1].text(j, i, f"{evidence[i, j]:.1f}{marker}", ha="center", va="center", fontsize=8, color="white" if evidence[i, j] > 8 else "black")
    fig.colorbar(images[0], ax=axes[0], shrink=.78, label="Probability difference")
    fig.colorbar(images[1], ax=axes[1], shrink=.78, label="−log10(BY q)")
    fig.suptitle("Does a completed noncanonical segment predict a later one?", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "recurrence_heatmaps.png", dpi=190)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_summary()
    segments = segment_summary()
    prompt_survival = prompt_survival_summary()
    recurrence = read_csv(RESULTS / "recurrence_all.csv")
    discoveries = [row for row in recurrence if float(row["by_q_value"]) < .05]
    write_csv(OUT / "checkpoint_summary.csv", checkpoints)
    write_csv(OUT / "segment_summary.csv", segments)
    write_csv(OUT / "prompt_survival_summary.csv", prompt_survival)
    write_csv(OUT / "adjusted_recurrence_discoveries.csv", discoveries)
    plot_overview(checkpoints)
    plot_yield(checkpoints)
    plot_segments(segments)
    plot_recurrence(recurrence)
    print(f"wrote {len(checkpoints)} checkpoint rows, {len(segments)} segment rows, and {len(discoveries)} adjusted recurrence discoveries")


if __name__ == "__main__":
    main()
