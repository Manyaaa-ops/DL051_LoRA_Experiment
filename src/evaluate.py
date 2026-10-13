"""Evaluation on the SST-2 validation split + results-table helpers."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

__all__ = ["evaluate", "format_results_table"]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str = "sst2",
) -> Dict[str, float]:
    """Compute mean cross-entropy loss and accuracy over ``loader``.

    For SST-2 the metric is plain accuracy on the 872-example GLUE
    validation split (the GLUE test labels are not public, so the
    validation split is used for both evaluation and model selection;
    this is stated as a limitation in the report).
    """
    was_training = model.training
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        total_loss += F.cross_entropy(out.logits, batch["label"], reduction="sum").item()
        correct += (out.logits.argmax(-1) == batch["label"]).sum().item()
        seen += batch["label"].numel()
    if was_training:
        model.train()
    key = "val" if task == "sst2" else task
    return {f"{key}_loss": total_loss / seen, f"{key}_acc": correct / seen}


def format_results_table(rows: List[Dict[str, Any]]) -> str:
    """Render a fixed-width comparison table from results.csv rows."""
    cols = [
        ("run_name", "run", "s"), ("method", "method", "s"),
        ("rank", "rank", "s"), ("best_val_acc", "val_acc", "f"),
        ("trainable_params", "trainable", "d"), ("trainable_pct", "t%", "f"),
        ("peak_gpu_mem_mib", "gpu_MiB", "f"), ("train_time_sec", "time_s", "f"),
    ]
    header = " | ".join(f"{h:>{w}}" if k != "s" else f"{h:<{w}}"
                        for _, h, k, w in
                        [(c, c[1], c[2], 10 if c[2] != "s" else 12) for c in cols])
    lines = [header, "-" * len(header)]
    for r in rows:
        cells = []
        for key, _, kind in cols:
            v = r.get(key, "")
            if v in ("", None) or (isinstance(v, str) and v.lower() == "nan"):
                v = "n/a"
            fmt = (f"{{:>{10}}}" if kind != "s" else f"{{:<{12}}}")
            if kind == "f" and isinstance(v, (int, float)):
                v = f"{float(v):.4f}" if key.endswith("acc") or key.endswith("pct") else f"{float(v):.1f}"
            cells.append(fmt.format(v))
        lines.append(" | ".join(cells))
    return "\n".join(lines)
