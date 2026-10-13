"""SST-2 (GLUE) data loading, preprocessing and label mapping.

Dataset
-------
GLUE SST-2 (Socher et al., 2013; Wang et al., 2018), loaded from the Hugging
Face Hub mirror ``nyu-mll/glue`` (Parquet files under the ``sst2`` config).

Splits
------
* train      : 67,349 labelled sentences
* validation :   872 labelled sentences  <- used for evaluation and model
                selection in this project (the GLUE *test* split has no
                public labels, so it cannot be scored locally).
* test       :   1,821 sentences, labels withheld.

Label mapping
-------------
0 = negative sentiment, 1 = positive sentiment (GLUE SST-2 convention).

Preprocessing
-------------
WordPiece tokenisation with the checkpoint's own tokenizer:
``truncation=True`` and ``padding='max_length'`` to ``max_len`` (128 by
default, as in the paper's RoBERTa experiments' sequence length), plus the
``attention_mask``. Inputs are returned as plain torch tensors, so a fixed
training subset is *literally the same example indices* for every method:
the subset is drawn once from a seeded permutation and stored as JSON for
auditability.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .model import DEFAULT_MODEL_NAME, build_tokenizer

__all__ = ["load_sst2", "save_subset_indices"]

HF_DATASET = "nyu-mll/glue"
HF_CONFIG = "sst2"
LABEL_NAMES = {0: "negative", 1: "positive"}


def save_subset_indices(indices: np.ndarray, path: str) -> None:
    """Persist the exact training-subset indices for auditability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"dataset": "nyu-mll/glue", "config": "sst2",
                   "split": "train", "n": int(len(indices)),
                   "indices": [int(i) for i in indices]}, f)


def load_sst2(
    model_name: str = DEFAULT_MODEL_NAME,
    max_len: int = 128,
    subset_size: Optional[int] = None,
    seed: int = 42,
    subset_index_path: Optional[str] = None,
) -> Dict[str, Dataset]:
    """Load and tokenise GLUE SST-2; return torch-format datasets.

    Parameters
    ----------
    subset_size:
        Number of training examples to use (None = full 67,349). The subset
        is a seeded random sample WITHOUT replacement; if
        ``subset_index_path`` exists, its stored indices are reused instead,
        guaranteeing the identical subset across experimental conditions.
    subset_index_path:
        Optional JSON path used to save/load the subset indices.
    """
    tokenizer = build_tokenizer(model_name)
    raw = load_dataset(HF_DATASET, HF_CONFIG)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    tok = raw.map(tokenize, batched=True, remove_columns=["sentence"])
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train = tok["train"]
    n_full = len(train)
    if subset_size is not None and subset_size < n_full:
        rng = np.random.RandomState(seed)
        reused = False
        if subset_index_path and os.path.exists(subset_index_path):
            with open(subset_index_path) as f:
                stored = json.load(f)
            assert stored["n"] == subset_size, (
                f"stored subset has {stored['n']} indices but subset_size="
                f"{subset_size}; delete the file or match the sizes"
            )
            indices = np.array(stored["indices"], dtype=np.int64)
            assert len(indices) == subset_size
            reused = True
        else:
            indices = rng.permutation(n_full)[:subset_size]
            if subset_index_path:
                save_subset_indices(indices, subset_index_path)
        train = train.select(indices.tolist())
        print(
            f"[data] training subset: {len(train)}/{n_full} examples "
            f"(seed={seed}, reused_from_file={reused})"
        )
    else:
        print(f"[data] full training split: {n_full} examples")

    return {
        "train": train,
        "validation": tok["validation"],
        "n_full_train": n_full,
        "n_train": len(train),
        "n_validation": len(tok["validation"]),
        "max_len": max_len,
        "label_names": LABEL_NAMES,
        "source": "GLUE SST-2 via Hugging Face (nyu-mll/glue)",
    }
