#!/usr/bin/env bash
# Run the full experiment suite sequentially:
#   1x full fine-tuning baseline + LoRA ranks 2, 4, 8
# Device is chosen automatically (CUDA on Colab T4, MPS on Apple silicon, else CPU).
# Identical subset (seeded, audited in results/data/subset_indices.json),
# identical epochs/batch/len/seed for every condition.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
SUBSET=${SUBSET:-8000}
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-16}

$PY -m src.train --method full --epochs "$EPOCHS" --subset "$SUBSET" --batch "$BATCH"
for R in 2 4 8; do
  $PY -m src.train --method lora --rank "$R" --epochs "$EPOCHS" --subset "$SUBSET" --batch "$BATCH"
done
echo "All experiments finished. See results/results.csv"
