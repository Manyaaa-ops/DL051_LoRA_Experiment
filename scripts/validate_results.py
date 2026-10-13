"""Post-run validation of the experiment suite (assignment section 9).

Checks, against the ACTUAL logs on disk:
  1. all four runs (full, lora r=2/4/8) exist in results/results.csv
  2. no missing/invalid metrics (val acc in (0,1), params > 0, time > 0)
  3. identical data subset, seed, epochs, batch size, max_len across runs
  4. results.csv best_val_acc equals max val_acc in experiment_log.csv
  5. LoRA runs have frozen_backbone_verified truthy; lora params match 18,432*r
  6. subset indices file exists and matches the recorded subset size

Usage:  .venv/bin/python scripts/validate_results.py
Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["full", "lora_r2_a4", "lora_r4_a8", "lora_r8_a16"]
LORA_PARAMS = {"lora_r2_a4": 18432 * 2, "lora_r4_a8": 18432 * 4, "lora_r8_a16": 18432 * 8}

fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    res_path = os.path.join(ROOT, "results", "results.csv")
    log_path = os.path.join(ROOT, "results", "experiment_log.csv")
    df = pd.read_csv(res_path).drop_duplicates(subset=["run_name"], keep="last")
    log = pd.read_csv(log_path)
    d = df.set_index("run_name")

    print("[validate] completeness")
    check("all 4 runs present", set(RUNS) <= set(df["run_name"]),
          f"missing={set(RUNS) - set(df['run_name'])}")

    print("[validate] metric sanity")
    for r in RUNS:
        if r not in d.index:
            continue
        row = d.loc[r]
        acc = float(row["best_val_acc"])
        check(f"{r}: val acc in (0,1)", 0 < acc < 1, f"acc={acc}")
        check(f"{r}: params > 0", int(row["trainable_params"]) > 0)
        check(f"{r}: time > 0", float(row["train_time_sec"]) > 0)

    print("[validate] identical protocol across runs")
    sub = d.loc[[r for r in RUNS if r in d.index]]
    for col, expect in [("subset_size", None), ("epochs", None), ("batch_size", None),
                        ("max_len", None), ("seed", None), ("dataset", None)]:
        vals = sub[col].astype(str).unique()
        check(f"{col} identical across runs", len(vals) == 1, f"values={vals.tolist()}")

    print("[validate] results.csv vs experiment_log.csv")
    for r in RUNS:
        if r not in d.index:
            continue
        epochs = log[log["run_name"] == r]
        check(f"{r}: epochs logged", len(epochs) > 0, f"n={len(epochs)}")
        best_from_log = epochs["val_acc"].astype(float).max()
        check(f"{r}: best_val_acc matches log",
              abs(best_from_log - float(d.loc[r, "best_val_acc"])) < 1e-4,
              f"csv={d.loc[r, 'best_val_acc']} log={best_from_log:.4f}")

    print("[validate] LoRA specifics")
    for r, expected in LORA_PARAMS.items():
        if r not in d.index:
            continue
        check(f"{r}: trainable = lora + head",
              int(d.loc[r, "lora_params"]) == expected,
              f"lora={int(d.loc[r,'lora_params'])} expected={expected}")
        check(f"{r}: rank recorded", str(d.loc[r, "rank"]).startswith(str(expected // 18432)))
        check(f"{r}: backbone frozen verified",
              str(d.loc[r, "frozen_backbone_verified"]).lower() in ("true", "1"),
              f"value={d.loc[r,'frozen_backbone_verified']}")

    print("[validate] auditability artifacts")
    idx_path = os.path.join(ROOT, "results", "data", "subset_indices.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        check("subset index size matches", idx["n"] == int(d.loc[RUNS[0], "subset_size"]),
              f"idx={idx['n']}")
        check("subset indices unique", len(set(idx["indices"])) == idx["n"])
    else:
        check("subset index file exists", False, idx_path)
    for r in RUNS:
        if r not in d.index:
            continue
        check(f"{r}: config.json saved",
              os.path.exists(os.path.join(ROOT, "results", "runs", r, "config.json")))

    # ---- optional cross-device (Colab T4) confirmation ------------------
    t4_path = os.path.join(ROOT, "results", "colab_t4", "results.csv")
    if os.path.exists(t4_path):
        print("[validate] Colab T4 confirmation run")
        t4 = pd.read_csv(t4_path).drop_duplicates(subset=["run_name"], keep="last")
        t = t4.set_index("run_name")
        check("t4: all 4 runs present", set(RUNS) <= set(t.index),
              f"missing={set(RUNS) - set(t.index)}")
        check("t4: device is cuda", set(t4["device"].astype(str)) == {"cuda"},
              f"values={set(t4['device'])}")
        check("t4: protocol matches main run",
              int(t4["subset_size"].iloc[0]) == int(d.loc[RUNS[0], "subset_size"])
              and int(t4["epochs"].iloc[0]) == int(d.loc[RUNS[0], "epochs"]),
              f"subset={t4['subset_size'].iloc[0]} epochs={t4['epochs'].iloc[0]}")
        t4log_path = os.path.join(ROOT, "results", "colab_t4", "experiment_log.csv")
        t4log = pd.read_csv(t4log_path) if os.path.exists(t4log_path) else None
        for r in RUNS:
            if r not in t.index:
                continue
            acc = float(t.loc[r, "best_val_acc"])
            check(f"t4 {r}: val acc in (0,1)", 0 < acc < 1, f"acc={acc}")
            check(f"t4 {r}: params match main run",
                  int(t.loc[r, "trainable_params"]) == int(d.loc[r, "trainable_params"]))
            if t4log is not None:
                best = t4log[t4log["run_name"] == r]["val_acc"].astype(float).max()
                check(f"t4 {r}: best_val_acc matches log",
                      abs(best - acc) < 1e-4, f"csv={acc} log={best:.4f}")

    print(f"[validate] {len(fails)} failure(s)" if fails else "[validate] ALL CHECKS PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
