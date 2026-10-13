"""Model construction, LoRA adaptation and parameter accounting.

Honesty note: the LoRA paper (Hu et al., ICLR 2022) reports results for
RoBERTa, DeBERTa, GPT-2 and GPT-3. It does NOT report DistilBERT results.
Our DistilBERT + SST-2 experiments are a reduced-scale adaptation of the
paper's *principle*, not a reproduction of a specific paper row, and the
report states this explicitly.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .lora import (
    LoRALinear,
    inject_lora,
    lora_parameter_names,
    mark_only_lora_and_head_trainable,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "build_model",
    "build_tokenizer",
    "apply_lora",
    "parameter_report",
    "lora_state_dict",
    "adapted_module_names",
]

DEFAULT_MODEL_NAME = "distilbert-base-uncased"

# Classification-head parameter name prefixes for DistilBERT sequence
# classification models ("pre_classifier" + "classifier"), plus the generic
# BERT-style head names so the helper also works on other checkpoints.
HEAD_PREFIXES: Tuple[str, ...] = ("classifier", "pre_classifier", "score")


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 2,
    dropout: float = 0.1,
):
    """Load a FRESH pretrained checkpoint for sequence classification.

    Every call downloads/reads the same pretrained weights, so each
    experimental run starts from identical initial *backbone* weights
    (head weights are randomly initialised per run by design; this is
    standard practice and identical across all compared methods).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        dropout=dropout,
    )
    return model


def build_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Load the WordPiece tokenizer matching the pretrained checkpoint."""
    return AutoTokenizer.from_pretrained(model_name)


def apply_lora(
    model: nn.Module,
    r: int = 4,
    alpha: Optional[float] = None,
    targets: Sequence[str] = ("q_lin", "v_lin"),
    lora_dropout: float = 0.0,
) -> List[str]:
    """Adapt ``targets`` projections of ``model`` with LoRA, in place.

    * Freezes every backbone parameter (done inside ``inject_lora``).
    * Wraps each matched ``nn.Linear`` in :class:`LoRALinear` (A random, B zero).
    * Re-enables the classification head so it remains trainable, as required
      by the assignment ("the classification head remains trainable").

    Returns the list of fully-qualified adapted module names (e.g.
    ``distilbert.transformer.layer.0.attention.q_lin``).
    """
    adapted = inject_lora(
        model,
        target_names=targets,
        r=r,
        alpha=alpha,
        lora_dropout=lora_dropout,
        freeze_backbone=True,
    )
    mark_only_lora_and_head_trainable(model, head_prefixes=HEAD_PREFIXES)
    return adapted


def adapted_module_names(model: nn.Module) -> List[str]:
    """Names of all submodules that are currently ``LoRALinear`` wrappers."""
    return [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def parameter_report(model: nn.Module) -> Dict[str, float]:
    """Parameter accounting for the actual ``model`` object.

    ``total`` counts every parameter of the full model *including* the LoRA
    adapter parameters and the classification head. ``trainable`` counts
    parameters with ``requires_grad=True`` (adapters + head for LoRA runs;
    everything for full fine-tuning).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(
        p.numel() for n, p in model.named_parameters()
        if n.endswith("lora_A") or n.endswith("lora_B")
    )
    head = sum(
        p.numel() for n, p in model.named_parameters()
        if n.split(".")[0] in HEAD_PREFIXES
        or any(n == pref or n.startswith(pref + ".") for pref in HEAD_PREFIXES)
    )
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "trainable_pct": 100.0 * trainable / total,
        "lora_params": int(lora),
        "head_params": int(head),
    }


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return ONLY the adapter weights (+ head), i.e. the task-specific
    checkpoint to store per task. This mirrors the paper's claim that the
    stored task-specific update shrinks to tens of megabytes while the
    frozen backbone is shared across tasks."""
    keep_prefixes = tuple(HEAD_PREFIXES)
    sd = model.state_dict()
    return {
        k: v
        for k, v in sd.items()
        if k.endswith("lora_A")
        or k.endswith("lora_B")
        or k.split(".")[0] in keep_prefixes
    }
