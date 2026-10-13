# DL-051 — LoRA: Low-Rank Adaptation of Large Language Models
### A reduced-scale reproduction and rank ablation (DistilBERT + GLUE SST-2)

**Student:** Manya Dixit (24CS2026) · **Programme:** IDD · **Assignment:** DL-051
**Paper:** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — *LoRA: Low-Rank
Adaptation of Large Language Models*, ICLR 2022 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## Objective

Implement the paper's core contribution **from scratch in PyTorch** (a custom
`LoRALinear` layer — **no PEFT / adapter library**, enforced by a unit test),
integrate it into a pretrained DistilBERT for SST-2 sentiment classification,
and compare against full fine-tuning on accuracy, trainable parameters, peak
GPU memory and training time, with **LoRA rank r ∈ {2, 4, 8} as the primary
ablation**.

The adapted layer computes (paper Eq. 3 with scaling):

```
h = W0 x + (alpha / r) · B A x        W0 frozen · A ~ random · B = 0
```

## Repository layout

```
DL051-LoRA/
├── README.md                       ← this file
├── requirements.txt
├── configs/default.yaml            # experiment configuration
├── src/
│   ├── lora.py                     # custom LoRALinear + injection (core contribution)
│   ├── model.py                    # model building, LoRA application, parameter accounting
│   ├── data.py                     # GLUE SST-2 loading, tokenisation, auditable subset
│   ├── train.py                    # experiment runner (CLI + importable)
│   ├── evaluate.py                 # validation metrics + results formatting
│   └── utils.py                    # seeding, device, memory meter, CSV/JSON helpers
├── tests/test_lora.py              # 16 unit tests
├── scripts/
│   ├── smoke_test.py               # 11 pre-flight checks on the real model
│   ├── run_all.sh                  # runs full FT + LoRA r=2/4/8 sequentially
│   ├── make_plots.py               # all figures from results.csv
│   ├── validate_results.py         # post-run audit (completeness/consistency)
│   ├── report_tables.py            # generates LaTeX tables from measured logs
│   └── build_notebook.py           # regenerates the notebook
├── notebooks/DL051_LoRA_Experiment.ipynb   # runnable Colab notebook
├── results/
│   ├── results.csv                 # one summary row per run (measured)
│   ├── experiment_log.csv          # per-epoch losses/accuracies (measured)
│   ├── data/subset_indices.json    # exact training-subset indices (audit)
│   ├── plots/*.png                 # figures used in report & slides
│   └── runs/<run>/config.json      # full config snapshot per run
├── report/report.tex · references.bib · report.pdf   # IEEE 2-column report
└── presentation/slides.tex · slides.pdf · viva_prep.md
```

## Method in one paragraph

Full fine-tuning updates all `d·k` weights of every matrix per task and needs
gradients + two Adam buffers for each, plus a full model copy per task. LoRA
freezes the pretrained weight `W0` and learns a rank-`r` update `ΔW = (α/r)·BA`
with `B ∈ ℝ^{d×r}` zero-initialised and `A ∈ ℝ^{r×k}` randomly initialised —
so training starts exactly at the pretrained function, only `r(d+k)` parameters
train per adapted layer, and the update merges back into `W0` for deployment
(no added inference latency). We follow the paper in adapting the **query and
value projections** of every attention block (12 of DistilBERT's linears), keep
the **classification head trainable**, and use the documented rule **α = 2r**
(so α/r = 2 is constant across ranks, isolating rank as the ablation variable).

## Setup

### Fresh local environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab (T4 GPU)

1. Upload/clone this repository into Colab (e.g. `!git clone <REPO_URL> DL051-LoRA && %cd DL051-LoRA`),
2. `!pip install -r requirements.txt`,
3. open `notebooks/DL051_LoRA_Experiment.ipynb` and **Runtime → Run all**
   (the notebook detects CUDA and records peak memory with
   `torch.cuda.max_memory_allocated`), or run `!bash scripts/run_all.sh` for the
   exact same protocol as the local runs below.

## How to run

```bash
# 1. unit tests (16) — no downloads, seconds
pytest -q tests/test_lora.py

# 2. pre-flight smoke test on the real checkpoint (11 checks)
python scripts/smoke_test.py --device auto

# 3. the full experiment suite (full FT + LoRA r=2/4/8)
bash scripts/run_all.sh                       # ~35 min on Apple M-series GPU,
                                              # longer on CPU, ~20 min on T4
# or a single run:
python -m src.train --method lora --rank 4 --epochs 2 --subset 8000 --batch 16

# 4. figures + audit + report tables
python scripts/make_plots.py
python scripts/validate_results.py
python scripts/report_tables.py
```

Every run appends to `results/results.csv` and `results/experiment_log.csv`
and stores its config in `results/runs/<run>/config.json`. The training subset
(8,000 of 67,349 examples, seed 42) is drawn **once** and shared literally by
all conditions (`results/data/subset_indices.json`).

## Results (measured in this project — not from the paper)

Device: Apple-silicon GPU (MPS). Model `distilbert-base-uncased`, GLUE SST-2
validation accuracy, subset 8,000, 2 epochs, batch 16, seq len 128, seed 42,
fp32, AdamW. Peak memory = allocator statistics (sampled per batch on MPS).

| Run | Total params | Trainable params | Trainable % | Best val acc | Peak mem (MiB) | Train time (min) |
|---|---:|---:|---:|---:|---:|---:|
| Full fine-tuning | 66,955,010 | 66,955,010 | 100.000 | **88.65 %** | 1027 | 10.6 |
| LoRA r=2 (α=4)  | 66,991,874 | 628,994 | 0.939 | 87.27 % | 264 | 5.9 |
| LoRA r=4 (α=8)  | 67,028,738 | 665,858 | 0.993 | **87.84 %** | 265 | 6.0 |
| LoRA r=8 (α=16) | 67,102,466 | 739,586 | 1.102 | 87.73 % | 266 | 6.0 |

* LoRA r=4 trains **~100× fewer parameters** (0.99 % of the model) for
  **0.81 points** less validation accuracy than full fine-tuning in this
  single-seed setting.
* **Rank ablation:** adapter parameters scale exactly linearly
  (18,432·r → 36,864 / 73,728 / 147,456), while accuracy saturates
  (87.27 → 87.84 → 87.73 for r=2/4/8): consistent with the paper's finding
  that small ranks suffice when adapting W_q and W_v.
* **Memory:** peak allocated memory 1027 MiB (full FT) vs ~265 MiB (LoRA) —
  LoRA removes gradient+Adam state for 66M parameters, but the frozen
  backbone and activations dominate, so the drop is ~4×, not ~100×.
* All LoRA runs verified bit-identical frozen backbone weights after training.

## Comparison with the paper (distinctly different settings!)

The paper reports, e.g., **RoBERTa-large on GLUE**: 355.0M full-FT parameters
vs **0.8M LoRA** (r=4, W_q+W_v), SST-2 **96.4 vs 96.2 ± 0.5**; and **GPT-3 175B**:
LoRA with 4.7M parameters matching/beating 175,255.8M full-FT parameters
(WikiSQL 73.4 vs 73.8; MNLI-m 91.7 vs 89.5), 1.2 TB → 350 GB training memory,
350 GB → 35 MB checkpoints, +25 % throughput. **Our numbers are not directly
comparable** (DistilBERT-66M, ≤8k examples, 2 epochs, single seed, Apple GPU);
we reproduce the *qualitative* claims — near-parity accuracy at ~100× fewer
trainable parameters, saturating returns in rank — at reduced scale. The paper
never reports DistilBERT experiments; this is stated in the report.

## Limitations

* Single seed; a 8,000-example training subset; 2 epochs. Differences of
  ±0.5 points between ranks are within noise — do not over-interpret them.
* The validation split is used for both model selection and reporting
  (GLUE test labels are withheld).
* Local measurements are on Apple MPS, not the Colab T4; the notebook
  reproduces the identical protocol on CUDA.
* Optional secondary ablations (α rule, adapter placement) were not run.

## Reproducibility checklist

* Seeds fixed (42); subset indices persisted and shared across runs.
* Fresh pretrained checkpoint per run; identical hyperparameters per condition.
* `results.csv` values are validated against `experiment_log.csv` by
  `scripts/validate_results.py`; all 37 checks pass.
* Report tables are **generated** from the logs by `scripts/report_tables.py` —
  no hand-typed numbers anywhere.
* 16 unit tests + 11 smoke checks pass before every experiment session.

## Citation

```bibtex
@inproceedings{hu2022lora,
  title     = {Lo{RA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan
               and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {ICLR},
  year      = {2022},
  note      = {arXiv:2106.09685}
}
```

No third-party LoRA implementation (PEFT, official repo) was consulted or used
for the implementation; standard components only (`nn.Linear`, AdamW,
`transformers` tokenizer/checkpoint, `datasets` for GLUE SST-2).
