"""Pre-flight sanity checks for DL-051 (run before any long experiment).

Verifies on a REAL DistilBERT + a tiny SST-2 batch:
  1. LoRA layer math (shape, zero-init identity, merged-weight equivalence)
  2. Forward/backward with labels and attention masks
  3. LoRA parameters receive gradients
  4. Frozen backbone weights are unchanged by an optimizer step
  5. Classification head is trainable
  6. Trainable parameter count matches the r * (d_in + d_out) formula

Usage:  .venv/bin/python scripts/smoke_test.py [--device auto]
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_sst2                      # noqa: E402
from src.lora import LoRALinear                     # noqa: E402
from src.model import apply_lora, build_model, parameter_report  # noqa: E402
from src.utils import resolve_device, set_seed     # noqa: E402

CHECKS = []


def check(name: str, ok: bool) -> None:
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="distilbert-base-uncased")
    args = ap.parse_args()
    dev = resolve_device(args.device)
    set_seed(42)
    print(f"[smoke] device={dev}")

    # 1) layer math -------------------------------------------------------
    base = torch.nn.Linear(32, 48)
    m = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(5, 32)
    check("LoRALinear output shape", m(x).shape == (5, 48))
    check("zero-init identity at start", torch.allclose(m(x), base(x), atol=1e-6))
    with torch.no_grad():
        m.lora_B.add_(torch.randn(48, 4) * 0.1)
    merged = torch.nn.functional.linear(x, m.merged_weight(), base.bias)
    check("forward == merged W+(a/r)BA", torch.allclose(m(x), merged, atol=1e-5))

    # 2-6) real model, real data batch ------------------------------------
    model = build_model(args.model)
    n_layers = model.config.n_layers
    adapted = apply_lora(model, r=4, alpha=8.0)
    model.to(dev)
    rep = parameter_report(model)
    expected_lora = n_layers * 2 * 4 * (model.config.dim * 2)
    print(f"[smoke] adapted {len(adapted)} modules; trainable={rep['trainable_params']:,} "
          f"({rep['trainable_pct']:.3f}% of {rep['total_params']:,})")
    check("adapters in q_lin+v_lin of every layer", len(adapted) == n_layers * 2)
    check("trainable = lora + head", rep["trainable_params"] == rep["lora_params"] + rep["head_params"])
    check("param-count formula r*(din+dout)", rep["lora_params"] == expected_lora)

    head_trains = all(
        p.requires_grad
        for n, p in model.named_parameters()
        if n.split(".")[0] in ("classifier", "pre_classifier")
    )
    check("classification head trainable", head_trains)
    check(
        "backbone frozen",
        all(
            not p.requires_grad
            for n, p in model.named_parameters()
            if not (n.split(".")[0] in ("classifier", "pre_classifier")
                    or n.endswith(("lora_A", "lora_B")))
        ),
    )

    probe = model.get_submodule(
        f"{model.base_model_prefix}.transformer.layer.0.attention.q_lin"
    ).base_layer.weight.detach().cpu().clone()

    data = load_sst2(args.model, max_len=128, subset_size=32, seed=42)
    batch_ids, batch_mask, batch_y = [], [], []
    for i in range(4):
        row = data["train"][i]
        batch_ids.append(row["input_ids"])
        batch_mask.append(row["attention_mask"])
        batch_y.append(row["label"])
    ids = torch.stack(batch_ids).to(dev)
    mask = torch.stack(batch_mask).to(dev)
    labels = torch.stack(batch_y).to(dev)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if n.endswith("lora_B")]
    check("forward+backward runs (loss finite)", torch.isfinite(out.loss))
    check("LoRA B receives gradients", all(g is not None for g in grads)
          and max(g.abs().sum() for g in grads) > 0)
    opt.step()

    cur = model.get_submodule(
        f"{model.base_model_prefix}.transformer.layer.0.attention.q_lin"
    ).base_layer.weight
    check("frozen weight unchanged after optimizer step",
          torch.equal(cur.detach().cpu(), probe))

    n_fail = sum(1 for _, ok in CHECKS if not ok)
    print(f"[smoke] {len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
