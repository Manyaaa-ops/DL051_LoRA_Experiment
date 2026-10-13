"""Training / experiment runner for DL-051.

One process = one experiment. Every run:

* builds a FRESH pretrained checkpoint (identical starting backbone weights),
* uses the SAME seeded training-subset indices and the same evaluation method,
* trains with AdamW over trainable parameters only,
* evaluates on the SST-2 validation split after every epoch and keeps the
  best checkpoint (by validation accuracy) for reporting,
* records wall-clock training time and (on CUDA devices only) peak allocated
  GPU memory; CPU runs record ``peak_gpu_mem_mib = nan`` instead of guessing,
* appends one summary row to ``results/results.csv`` and one row per epoch to
  ``results/experiment_log.csv``, and stores the exact run configuration in
  ``results/runs/<run_name>/config.json``.

Learning rates (documented defaults, not claimed optimal):
    full fine-tuning : 2e-5  (standard for transformer full FT)
    LoRA             : 1e-3  (adapters train from scratch; paper also uses
                              much larger LR for A/B than for the head)
Optimizer: AdamW, weight_decay=0.01, constant LR (no scheduler), gradient
clipping at 1.0.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import load_sst2
from .evaluate import evaluate
from .model import apply_lora, build_model, lora_state_dict, parameter_report
from .utils import (
    PeakMemoryMeter,
    append_csv_row,
    load_config,
    resolve_device,
    save_json,
    set_seed,
)

__all__ = ["run_experiment", "DEFAULT_LR"]

DEFAULT_LR = {"full": 2e-5, "lora": 1e-3}


def _find_probe_param(model: torch.nn.Module):
    """A small, purely-backbone parameter used to VERIFY freezing.

    Prefer layer-0 attention query weight (~2.4 MB fp32): present in every
    run, tiny compared with the embedding matrix.
    """
    name = "distilbert.transformer.layer.0.attention.q_lin"
    mod = model.get_submodule(name)
    return getattr(mod, "base_layer", mod).weight


@torch.no_grad()
def _frozen_changed(model: torch.nn.Module, probe: torch.Tensor) -> bool:
    """True if the probed frozen weight differs from ``probe``."""
    current = _find_probe_param(model)
    return not torch.equal(current.detach().cpu(), probe)


def run_experiment(
    method: str = "lora",
    rank: int = 4,
    alpha: Optional[float] = None,
    epochs: int = 2,
    subset_size: Optional[int] = 8000,
    batch_size: int = 16,
    max_len: int = 128,
    lr: Optional[float] = None,
    weight_decay: float = 0.01,
    seed: int = 42,
    model_name: str = "distilbert-base-uncased",
    device: str = "auto",
    results_dir: str = "results",
    subset_index_path: Optional[str] = None,
    save_checkpoint: bool = True,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one training experiment and persist all artifacts/metrics."""
    assert method in ("full", "lora"), f"unknown method {method!r}"
    if alpha is None:
        alpha = 2.0 * rank          # project rule: alpha = 2r
    if lr is None:
        lr = DEFAULT_LR[method]

    if run_name is None:
        run_name = method if method == "full" else f"lora_r{rank}_a{int(alpha)}"
    if subset_index_path is None:
        subset_index_path = os.path.join(results_dir, "data", "subset_indices.json")

    dev = resolve_device(device)
    set_seed(seed)

    # ---- data (identical subset for every method, seeded & audited) ----
    data = load_sst2(
        model_name=model_name,
        max_len=max_len,
        subset_size=subset_size,
        seed=seed,
        subset_index_path=subset_index_path,
    )
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        data["train"], batch_size=batch_size, shuffle=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(data["validation"], batch_size=batch_size, num_workers=0)

    # ---- model (fresh pretrained weights every run) ----
    model = build_model(model_name)
    adapted = []
    if method == "lora":
        adapted = apply_lora(model, r=rank, alpha=alpha)
    params = parameter_report(model)
    model.to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    probe = _find_probe_param(model).detach().cpu().clone()  # freezing verification
    meter = PeakMemoryMeter(dev)
    meter.reset()  # peak stats restart from current allocations (incl. params)

    # ---- training ----
    history = []
    best_val_acc, best_epoch = -1.0, -1
    best_state = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses, epoch_correct, epoch_seen = [], 0, 0
        ep_start = time.time()
        for batch in train_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            meter.sample()  # MPS has no peak-stats API: sample per batch
            epoch_losses.append(loss.item())
            epoch_correct += (out.logits.argmax(-1) == batch["label"]).sum().item()
            epoch_seen += batch["label"].numel()

        train_acc = epoch_correct / epoch_seen
        train_loss = float(np.mean(epoch_losses))
        val = evaluate(model, val_loader, dev)
        ep_time = time.time() - ep_start

        row = {
            "run_name": run_name, "method": method,
            "rank": rank if method == "lora" else "",
            "alpha": alpha if method == "lora" else "",
            "epoch": epoch,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val["val_loss"], 4), "val_acc": round(val["val_acc"], 4),
            "epoch_time_sec": round(ep_time, 1),
        }
        history.append(row)
        print(
            f"[{run_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val['val_loss']:.4f} val_acc={val['val_acc']:.4f} "
            f"({ep_time:.1f}s)",
            flush=True,
        )
        if val["val_acc"] > best_val_acc:
            best_val_acc, best_epoch = val["val_acc"], epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    train_time = time.time() - t0
    peak_mem = meter.read()
    # For full fine-tuning the backbone is intentionally trained, so the
    # "unchanged" check only has meaning for LoRA runs.
    frozen_ok = (
        "n/a (full FT trains backbone)" if method == "full"
        else not _frozen_changed(model, probe)
    )

    # ---- artifacts ----
    run_dir = os.path.join(results_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path, adapter_path, adapter_bytes = "", "", 0
    if save_checkpoint:
        assert best_state is not None
        ckpt_path = os.path.join(run_dir, "best_model.pt")
        torch.save(best_state, ckpt_path)
        if method == "lora":
            adapter_path = os.path.join(run_dir, "lora_adapter.pt")
            torch.save(lora_state_dict(model), adapter_path)
            adapter_bytes = os.path.getsize(adapter_path)

    config = {
        "run_name": run_name, "method": method, "rank": rank if method == "lora" else None,
        "alpha": alpha if method == "lora" else None,
        "epochs": epochs, "subset_size": subset_size,
        "n_train_examples": data["n_train"], "n_full_train": data["n_full_train"],
        "n_validation_examples": data["n_validation"],
        "batch_size": batch_size, "max_len": max_len, "lr": lr,
        "weight_decay": weight_decay, "seed": seed, "model_name": model_name,
        "dataset": data["source"], "device": str(dev),
        "lora_targets": adapted or None, "lora_dropout": 0.0,
        "optimizer": "AdamW(constant lr, grad clip 1.0)",
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(config, os.path.join(run_dir, "config.json"))

    summary = {
        "run_name": run_name, "method": method,
        "rank": rank if method == "lora" else "",
        "alpha": alpha if method == "lora" else "",
        "best_val_acc": round(best_val_acc, 4), "best_epoch": best_epoch,
        "final_val_acc": history[-1]["val_acc"],
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "trainable_pct": round(params["trainable_pct"], 4),
        "lora_params": params["lora_params"],
        "train_time_sec": round(train_time, 1),
        "peak_gpu_mem_mib": ("" if peak_mem is None else round(peak_mem, 1)),
        "frozen_backbone_verified": frozen_ok,
        "subset_size": data["n_train"], "epochs": epochs,
        "batch_size": batch_size, "max_len": max_len, "lr": lr, "seed": seed,
        "device": str(dev), "model_name": model_name, "dataset": "GLUE SST-2 (nyu-mll/glue)",
        "adapter_checkpoint_bytes": adapter_bytes,
        "timestamp": config["timestamp"],
    }
    append_csv_row(os.path.join(results_dir, "results.csv"), summary)
    for r in history:
        append_csv_row(os.path.join(results_dir, "experiment_log.csv"), r)
    print(f"[{run_name}] DONE best_val_acc={best_val_acc:.4f} "
          f"(epoch {best_epoch}) trainable={params['trainable_params']:,} "
          f"({params['trainable_pct']:.3f}%) time={train_time:.0f}s "
          f"peak_gpu_mem={peak_mem} frozen_ok={frozen_ok}", flush=True)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DL-051 LoRA experiment runner")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--method", choices=["full", "lora"], default="lora")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--subset", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--from-config", action="store_true",
                    help="read epochs/subset/batch/max_len from the yaml config")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    kwargs: Dict[str, Any] = dict(
        method=args.method, rank=args.rank, alpha=args.alpha, epochs=args.epochs,
        subset_size=args.subset, batch_size=args.batch, max_len=args.max_len,
        lr=args.lr, seed=args.seed, model_name=args.model, device=args.device,
        results_dir=args.results_dir, run_name=args.run_name,
    )
    if args.from_config and os.path.exists(args.config):
        cfg = load_config(args.config).get("experiment", {})
        kwargs["subset_size"] = cfg.get("subset_size", kwargs["subset_size"])
        kwargs["epochs"] = cfg.get("epochs", kwargs["epochs"])
        kwargs["batch_size"] = cfg.get("batch_size", kwargs["batch_size"])
        kwargs["max_len"] = cfg.get("max_len", kwargs["max_len"])
    run_experiment(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
