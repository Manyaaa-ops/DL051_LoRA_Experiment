"""Generate the report figures from results/results.csv.

Only ACTUAL measurements are plotted: rows with an empty peak-memory field
(e.g. CPU runs) are excluded from the memory figure, and the figure is
skipped entirely if no valid measurement exists.

Usage:  .venv/bin/python scripts/make_plots.py [--results results/results.csv]
                                              [--out results/plots]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN_ORDER = ["full", "lora_r2_a4", "lora_r4_a8", "lora_r8_a16"]


def load(results_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    df["peak_gpu_mem_mib"] = pd.to_numeric(df["peak_gpu_mem_mib"], errors="coerce")
    df["run_name"] = df["run_name"].astype(str)
    order = {n: i for i, n in enumerate(RUN_ORDER)}
    df["_o"] = df["run_name"].map(lambda n: order.get(n, len(order)))
    return df.sort_values("_o").reset_index(drop=True)


def labels(df: pd.DataFrame) -> list[str]:
    return [
        f"{r.run_name}\n(val {float(r.best_val_acc)*100:.1f}%)"
        if r.method == "lora"
        else f"full FT\n(val {float(r.best_val_acc)*100:.1f}%)"
        for r in df.itertuples()
    ]


def plot_accuracy(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    acc = df["best_val_acc"].astype(float) * 100
    colors = ["#4C72B0" if m == "full" else "#55A868" for m in df["method"]]
    bars = ax.bar(labels(df), acc, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=2, fontsize=9)
    ax.set_ylabel("Best SST-2 validation accuracy (%)")
    ax.set_xlabel("Method (r = LoRA rank)")
    ax.set_title(
        "Full fine-tuning vs custom LoRA on DistilBERT-base-uncased, GLUE SST-2\n"
        f"(subset n={int(df['subset_size'].iloc[0])}, 2 epochs, seed 42, measured)")
    ax.set_ylim(min(acc) - 3, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)


def plot_params(df: pd.DataFrame, out: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    tp = df["trainable_params"].astype(float)
    colors = ["#4C72B0" if m == "full" else "#55A868" for m in df["method"]]
    b1 = ax1.bar(labels(df), tp / 1e6, color=colors)
    ax1.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Trainable parameters (millions, log scale)")
    ax1.set_title("Trainable parameters")

    pct = df["trainable_pct"].astype(float)
    b2 = ax2.bar(labels(df), pct, color=colors)
    ax2.bar_label(b2, fmt="%.3f%%", padding=2, fontsize=8)
    ax2.set_ylabel("Trainable parameters (% of total)")
    ax2.set_title("Fraction of the model being trained")
    ax2.set_yscale("log")
    fig.suptitle("Parameter efficiency of custom LoRA vs full fine-tuning (measured)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trainable_params.png"), dpi=150)
    plt.close(fig)


def plot_memory(df: pd.DataFrame, out: str) -> bool:
    mem = df.dropna(subset=["peak_gpu_mem_mib"])
    if mem.empty:
        print("[plots] no valid peak-memory measurements -> memory figure skipped")
        return False
    fig, ax = plt.subplots(figsize=(7, 4.2))
    m = mem["peak_gpu_mem_mib"].astype(float)
    colors = ["#4C72B0" if x == "full" else "#55A868" for x in mem["method"]]
    bars = ax.bar(labels(mem), m, color=colors)
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=9)
    device = mem["device"].iloc[0]
    ax.set_ylabel("Peak allocated device memory (MiB)")
    ax.set_xlabel("Method (r = LoRA rank)")
    ax.set_title(
        "Peak allocated GPU memory during training (measured)\n"
        f"device={device}, batch=16, seq_len=128, fp32 (allocator statistics)")
    ax.set_ylim(0, float(m.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "peak_memory.png"), dpi=150)
    plt.close(fig)
    return True


def plot_time(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = df["train_time_sec"].astype(float) / 60.0
    colors = ["#4C72B0" if m == "full" else "#55A868" for m in df["method"]]
    bars = ax.bar(labels(df), t, color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set_ylabel("Total training time (minutes, incl. per-epoch evaluation)")
    ax.set_xlabel("Method (r = LoRA rank)")
    ax.set_title(f"Measured training wall-clock time (device={df['device'].iloc[0]})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_time.png"), dpi=150)
    plt.close(fig)


def plot_acc_vs_params(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tp = df["trainable_params"].astype(float)
    acc = df["best_val_acc"].astype(float) * 100
    colors = ["#4C72B0" if m == "full" else "#55A868" for m in df["method"]]
    ax.scatter(tp, acc, c=colors, s=90, zorder=3)
    for row, name in zip(df.itertuples(), labels(df)):
        ax.annotate(row.run_name, (float(row.trainable_params), float(row.best_val_acc) * 100),
                    textcoords="offset points", xytext=(8, -3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Best SST-2 validation accuracy (%)")
    ax.set_title("Accuracy vs. trainable parameter budget (measured)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_vs_params.png"), dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.results)
    plot_accuracy(df, args.out)
    plot_params(df, args.out)
    has_mem = plot_memory(df, args.out)
    plot_time(df, args.out)
    plot_acc_vs_params(df, args.out)
    print(f"[plots] wrote figures to {args.out}/ (memory figure: {'yes' if has_mem else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
