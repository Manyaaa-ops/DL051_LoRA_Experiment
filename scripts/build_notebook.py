"""Generate notebooks/DL051_LoRA_Experiment.ipynb (executable, not a stub).

Modes (set before running):
  DL051_MODE=quick  -> tiny subset, 1 epoch  (~3 min on CPU/MPS; used to
                       verify the notebook actually executes)
  DL051_MODE=full   -> assignment-scale settings: subset 8000, 2 epochs,
                       ranks 2/4/8 + full-FT baseline (default; matches
                       the report's local experiments, and is the Colab
                       configuration)
Usage:  .venv/bin/python scripts/build_notebook.py
"""

from __future__ import annotations

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md("""# DL-051 — LoRA: Low-Rank Adaptation of Large Language Models
**Reduced-scale reproduction and rank ablation on DistilBERT + GLUE SST-2**

* Student: Manya Dixit (24CS2026), Programme: IDD
* Paper: Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — *LoRA: Low-Rank
  Adaptation of Large Language Models*, ICLR 2022 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))

**Objective.** Implement the core contribution of the paper — a custom low-rank
adapter layer — *from scratch in PyTorch* (no PEFT library), integrate it into a
pretrained DistilBERT for SST-2 sentiment classification, and compare against
full fine-tuning on accuracy, trainable parameters, peak GPU memory and training
time, with the **LoRA rank r ∈ {2, 4, 8} as the primary ablation**.

**Honesty note.** The original paper reports RoBERTa/GPT-2/GPT-3 results; it does
not report DistilBERT or SST-2-at-this-scale reproduction targets. This notebook
adapts the paper's *principle* at reduced scale and every number shown is produced
by an actual run in this session — nothing is pre-filled or fabricated."""))

cells.append(md("""## 1. Paper: the key idea and equation

Full fine-tuning updates every pretrained weight $W_0 \\in \\mathbb{R}^{d\\times k}$:
for a 175B model that means updating and storing ~175B task-specific numbers, plus
optimizer states for all of them during training.

The paper constrains the update to be **low rank**:

$$W_0 + \\Delta W = W_0 + BA, \\qquad B\\in\\mathbb{R}^{d\\times r},\\ A\\in\\mathbb{R}^{r\\times k},\\ r \\ll \\min(d,k)$$

so the forward pass of an adapted layer (paper Eq. 3 with scaling) is

$$h = W_0 x + \\frac{\\alpha}{r}\\,BAx$$

* $W_0$ is **frozen**; only $A, B$ (and the classification head) are trained.
* $B$ is initialised to **zero**, $A$ randomly (Gaussian) ⇒ $\\Delta W = 0$ at step 0,
  so training starts exactly at the pretrained function and learns a *change* to it.
* Trainable parameters per adapted layer drop from $d\\cdot k$ to $r(d+k)$.
* The update can be merged into $W_0$ for deployment ⇒ **no extra inference latency**.

This project follows the paper's choice of adapting the **query and value
projections** ($W_q, W_v$) of every attention block, with the documented rule
$\\alpha = 2r$ (so the scale factor $\\alpha/r = 2$ is constant across ranks)."""))

cells.append(code("""# 0) Environment setup -------------------------------------------------------
import os, sys, subprocess, time
print("python:", sys.version.split()[0])

# If running in a fresh Colab runtime, clone the repository first (fill in the
# repository URL after publishing), then re-run this cell:
#   !git clone <REPO_URL> DL051-LoRA && %cd DL051-LoRA

# Locate the repository root whether the kernel starts in the repo root or in
# notebooks/ (jupyter/nbconvert run the notebook with cwd = its own folder).
from pathlib import Path
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "src").is_dir() and (p / "requirements.txt").is_file()), None)
if ROOT is None:
    raise RuntimeError("Could not locate the repository root (folder with src/).")
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("repository root:", ROOT)

# Install dependencies (works on Colab; skipped gracefully in environments
# without pip where the requirements are already installed).
import importlib.util
if importlib.util.find_spec("pip") is not None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
else:
    print("pip unavailable in this environment - assuming requirements.txt is satisfied")

import torch
print("torch:", torch.__version__)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | {props.total_memory/1024**3:.1f} GiB")
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    print("GPU: Apple silicon (MPS)")
    DEVICE = "mps"
else:
    print("No GPU detected - CPU fallback (expect slower runs)")
    DEVICE = "cpu"
print("Using device:", DEVICE)"""))

cells.append(code("""# 1) Experiment configuration ------------------------------------------------
import dataclasses
MODE = os.environ.get("DL051_MODE", "full")   # "quick" = fast verification run

@dataclasses.dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 128            # paper's GLUE sequence length
    batch_size: int = 16
    seed: int = 42                # single seed (documented limitation)
    subset: int = 8000            # training examples of 67,349
    epochs: int = 2
    ranks: tuple = (2, 4, 8)      # primary ablation: rank
    alpha_rule: str = "alpha = 2 * r"
    lr_full: float = 2e-5
    lr_lora: float = 1e-3

cfg = Config()
if MODE == "quick":
    cfg.subset, cfg.epochs = 512, 1
print(cfg)

# Each notebook run appends to its own CSV so it never corrupts repo results.
RESULTS_DIR = f"results/notebook_{MODE}"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")"""))

cells.append(md("""## 2. Dataset: GLUE SST-2

* Source: GLUE SST-2 (Socher et al., 2013; Wang et al., 2018) via Hugging Face `nyu-mll/glue`.
* Splits: 67,349 train / 872 validation / 1,821 test (test labels withheld ⇒ we evaluate on **validation**).
* Preprocessing: WordPiece tokenisation with the checkpoint's own tokenizer, truncation + padding to 128, `attention_mask` kept.
* Label mapping: `0 = negative`, `1 = positive`.
* A seeded subset of the training split is drawn **once** and stored as JSON
  (`results/data/subset_indices.json`) so every method sees literally the same examples."""))

cells.append(code("""# 2) Data loading ------------------------------------------------------------
from src.data import load_sst2
from torch.utils.data import DataLoader

data = load_sst2(cfg.model_name, max_len=cfg.max_len, subset_size=cfg.subset,
                 seed=cfg.seed)
train_loader = DataLoader(data["train"], batch_size=cfg.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(cfg.seed))
val_loader = DataLoader(data["validation"], batch_size=cfg.batch_size)
print("train:", data["n_train"], "validation:", data["n_validation"],
      "labels:", data["label_names"])"""))

cells.append(md("""## 3. Custom LoRA implementation (from scratch)

`src/lora.py` implements `LoRALinear` using only standard PyTorch
(`nn.Linear`, `F.linear`, `nn.Parameter`). **No `peft` import anywhere** — a unit
test enforces this. The forward pass is exactly
`output = base_layer(x) + (alpha / r) * linear(linear(x, A), B)`,
with the base weights frozen, `A` randomly initialised and `B` zero-initialised."""))

cells.append(code("""# 3) Show and import the custom implementation --------------------------------
from src.lora import LoRALinear, inject_lora
from src.model import apply_lora, build_model, parameter_report

print(open("src/lora.py").read()[:2600], "\\n... [truncated - see src/lora.py]")"""))

cells.append(md("""## 4. Unit tests and sanity checks

Before any long run we verify: output shapes, zero-initialisation identity,
merged-weight equivalence $W' = W + (\\frac{\\alpha}{r})BA$, gradient flow to
$A$/$B$, frozen backbone weights, trainable-parameter formulas, and an
end-to-end step on the real model."""))

cells.append(code("""# 4a) Unit tests --------------------------------------------------------------
!{sys.executable} -m pytest -q tests/test_lora.py"""))

cells.append(code("""# 4b) Pre-flight smoke test on the real model --------------------------------
!{sys.executable} scripts/smoke_test.py --device {DEVICE}"""))

cells.append(md("""## 5–8. Experiments: full fine-tuning baseline, then LoRA r = 2, 4, 8

Same pretrained checkpoint, same subset, same epochs/batch/length/seed,
same evaluation (validation accuracy after every epoch, best checkpoint kept).
Only the method (and, for LoRA, the rank) changes. Wall-clock time and peak
allocated GPU memory (CUDA allocator statistics; sampled MPS allocations on
Apple silicon) are recorded per run."""))

for name, method, rank in [
    ("Baseline: FULL fine-tuning", "full", None),
    ("LoRA rank 2", "lora", 2),
    ("LoRA rank 4", "lora", 4),
    ("LoRA rank 8", "lora", 8),
]:
    rank_arg = f"rank={rank}," if rank is not None else ""
    run_name = "full" if method == "full" else f"lora_r{rank}_a{2*rank}"
    cells.append(md(f"### {name}"))
    cells.append(code(f"""# {name}
from src.train import run_experiment

summary_{run_name} = run_experiment(
    method="{method}",{rank_arg}
    epochs=cfg.epochs, subset_size=cfg.subset, batch_size=cfg.batch_size,
    max_len=cfg.max_len, seed=cfg.seed, model_name=cfg.model_name,
    device=DEVICE, results_dir=RESULTS_DIR, run_name="{run_name}",
)
summary_{run_name}"""))

cells.append(md("## 9. Results table (all measured in this session)"))

cells.append(code("""# 9) Results table ------------------------------------------------------------
import pandas as pd
from src.evaluate import format_results_table

df = pd.read_csv(CSV_PATH)
display_cols = ["run_name", "method", "rank", "best_val_acc", "total_params",
                "trainable_params", "trainable_pct", "peak_gpu_mem_mib",
                "train_time_sec", "frozen_backbone_verified"]
print(format_results_table(df[display_cols].to_dict("records")))
df[display_cols]"""))

cells.append(md("## 10. Accuracy and trainable-parameter comparison"))

cells.append(code("""# 10) Plots: accuracy + parameter efficiency ----------------------------------
import matplotlib.pyplot as plt

order = {n: i for i, n in enumerate(["full", "lora_r2_a4", "lora_r4_a8", "lora_r8_a16"])}
df["_o"] = df["run_name"].map(lambda n: order.get(n, 99))
d = df.sort_values("_o")
xlabels = [f"full FT" if r.method == "full" else f"LoRA r={int(r.rank)}" for r in d.itertuples()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
colors = ["#4C72B0" if m == "full" else "#55A868" for m in d["method"]]
b1 = ax1.bar(xlabels, d["best_val_acc"].astype(float) * 100, color=colors)
ax1.bar_label(b1, fmt="%.2f%%", padding=2, fontsize=9)
ax1.set_ylim(float(d["best_val_acc"].astype(float).min()) * 100 - 3, 100)
ax1.set_ylabel("Best SST-2 validation accuracy (%)")
ax1.set_title(f"Validation accuracy (mode={MODE}, subset={cfg.subset}, seed={cfg.seed})")

b2 = ax2.bar(xlabels, d["trainable_params"].astype(float), color=colors)
ax2.set_yscale("log")
ax2.bar_label(b2, fmt="%.3g", padding=2, fontsize=8)
ax2.set_ylabel("Trainable parameters (log scale)")
ax2.set_title("Trainable parameter budget")
plt.tight_layout(); plt.show()"""))

cells.append(md("## 11. Peak GPU memory and training time"))

cells.append(code("""# 11) Memory + wall-clock comparison ------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

mem = pd.to_numeric(d["peak_gpu_mem_mib"], errors="coerce")
if mem.notna().any():
    b = ax1.bar(xlabels, mem, color=colors)
    ax1.bar_label(b, fmt="%.0f", padding=2, fontsize=9)
    ax1.set_ylabel("Peak allocated device memory (MiB)")
    ax1.set_title(f"Peak memory during training (device={DEVICE})")
else:
    ax1.text(0.5, 0.5, "no GPU memory measurement\\n(CPU run)", ha="center", va="center")
    ax1.set_axis_off()

t = d["train_time_sec"].astype(float) / 60
b = ax2.bar(xlabels, t, color=colors)
ax2.bar_label(b, fmt="%.1f", padding=2, fontsize=9)
ax2.set_ylabel("Training time (minutes)")
ax2.set_title("Wall-clock time incl. per-epoch evaluation")
plt.tight_layout(); plt.show()

print("Note: LoRA cuts gradient/optimizer-state memory, but the frozen backbone "
      "must still be stored and processed, so parameter/activation memory remains.")"""))

cells.append(md("""## 12. Discussion and limitations

* **What the paper reports vs what we measure.** The paper's GLUE table
  (RoBERTa-large): full FT 96.4 vs LoRA r=4 96.2 ± 0.5 SST-2 accuracy, with
  0.8M trainable vs 355M. Our runs are a *different, smaller* setting
  (DistilBERT-66M, ≤8k training examples, 2 epochs, single seed): absolute
  accuracies are not comparable to the paper's; the *pattern* (LoRA close to
  full FT at a tiny fraction of trainable parameters, rank barely moving the
  needle beyond small r) is what we check.
* **Rank ablation.** Per adapted layer the adapter costs $r(d+k)$ parameters:
  linear in $r$. The paper found little benefit from large ranks for $W_q,W_v$
  (r=1 already competitive on GPT-3 tasks); we test whether accuracy saturates
  at small r on our setting.
* **Limitations.** Single seed; validation used for both evaluation and model
  selection (GLUE test labels are withheld); reduced training subset; local
  GPU is Apple MPS, not the Colab T4 the notebook targets; 2 epochs only.
  Peak memory on MPS is sampled per batch, on CUDA it is exact allocator peak.
* **Failure cases.** Small subsets make runs noisy: with 8k examples a ±0.5%
  accuracy difference between ranks is within seed noise and must not be
  over-interpreted."""))

cells.append(md("""## 13. Conclusion

The custom LoRA layer reproduces the paper's core claim at reduced scale: a
low-rank update on the query/value projections, with the backbone frozen,
trains orders of magnitude fewer parameters than full fine-tuning while staying
close in SST-2 validation accuracy, with zero extra inference cost after merging
$W' = W_0 + (\\frac{\\alpha}{r})BA$. Rank controls the capacity/parameter trade-off
linearly; beyond a small rank the accuracy gains saturate in our setting.

**References**
1. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. arXiv:2106.09685
2. Sanh et al. *DistilBERT, a distilled version of smaller, faster, lighter BERT*, NeurIPS-W 2019.
3. Socher et al. *Recursive Deep Models for Semantic Compositionality*, EMNLP 2013 (SST).
4. Wang et al. *GLUE: A Multi-Task Benchmark*, EMNLP-W 2018."""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notebooks", "DL051_LoRA_Experiment.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
