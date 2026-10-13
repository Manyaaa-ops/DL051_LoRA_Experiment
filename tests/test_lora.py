"""Unit tests for the custom LoRA implementation.

These tests use a TINY randomly-initialised DistilBERT (built from a small
``DistilBertConfig``) so they run offline in seconds and still exercise the
real module names (``...attention.q_lin`` / ``v_lin``) and the real
classification head names (``pre_classifier`` / ``classifier``).

Run with:  pytest -q tests/test_lora.py
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lora import (  # noqa: E402
    LoRALinear,
    inject_lora,
    lora_parameter_names,
    mark_only_lora_and_head_trainable,
)
from src.model import HEAD_PREFIXES, apply_lora, parameter_report  # noqa: E402

D_IN, D_OUT, R, ALPHA = 16, 24, 4, 8.0


def _base() -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(D_IN, D_OUT)


def _tiny_bert(n_layers: int = 2, hidden: int = 32):
    cfg = DistilBertConfig(
        vocab_size=100, dim=hidden, n_layers=n_layers, n_heads=2,
        hidden_dim=hidden * 4, max_position_embeddings=64, num_labels=2,
    )
    torch.manual_seed(0)
    return DistilBertForSequenceClassification(cfg)


# ---------------------------------------------------------------------------
# LoRALinear: shapes, initialisation, math
# ---------------------------------------------------------------------------
def test_forward_shape_batched_and_unbatched():
    m = LoRALinear(_base(), r=R, alpha=ALPHA)
    x3 = torch.randn(2, 7, D_IN)
    assert m(x3).shape == (2, 7, D_OUT)
    x1 = torch.randn(5, D_IN)
    assert m(x1).shape == (5, D_OUT)


def test_b_zero_init_implies_identity_at_start():
    """B = 0 must make the adapted layer exactly reproduce the base layer."""
    m = LoRALinear(_base(), r=R, alpha=ALPHA)
    assert torch.count_nonzero(m.lora_B) == 0, "B must be initialised to zero"
    assert torch.count_nonzero(m.lora_A) > 0, "A must NOT be zero-initialised"
    x = torch.randn(3, 11, D_IN)
    assert torch.allclose(m(x), m.base_layer(x), atol=1e-6)


def test_forward_equals_merged_weight_formula():
    """forward(x) must equal F.linear(x, W + (alpha/r) B A, b) exactly."""
    base = _base()
    m = LoRALinear(base, r=R, alpha=ALPHA)
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(D_OUT, R))  # make the update non-trivial
    x = torch.randn(4, D_IN)
    expected = F.linear(x, m.merged_weight(), base.bias)
    assert torch.allclose(m(x), expected, atol=1e-5)


def test_scaling_is_alpha_over_r():
    m = LoRALinear(_base(), r=R, alpha=ALPHA)
    assert m.scaling == pytest.approx(ALPHA / R)


def test_rank_validation():
    base = _base()
    with pytest.raises(ValueError):
        LoRALinear(base, r=0, alpha=ALPHA)
    with pytest.raises(ValueError):
        LoRALinear(base, r=D_IN + 1, alpha=ALPHA)
    with pytest.raises(TypeError):
        LoRALinear(torch.nn.ReLU(), r=R, alpha=ALPHA)


# ---------------------------------------------------------------------------
# Gradients and freezing
# ---------------------------------------------------------------------------
def test_gradients_reach_adapters_only():
    m = LoRALinear(_base(), r=R, alpha=ALPHA)
    x = torch.randn(8, D_IN)
    m(x).sum().backward()
    # At step 0 B = 0, so dL/dA = scale * (G B)^T = 0: A's gradient tensor
    # exists but is exactly zero. That is expected mathematics, not a bug.
    assert m.lora_A.grad is not None
    assert torch.count_nonzero(m.lora_A.grad) == 0
    # B's gradient is generally non-zero at step 0 because it multiplies A.
    assert m.lora_B.grad is not None and m.lora_B.grad.abs().sum() > 0
    # After a B update, A receives non-zero gradients: 
    with torch.no_grad():
        m.lora_B.add_(0.1 * torch.randn(D_OUT, R))
    m.zero_grad()
    m(x).sum().backward()
    assert m.lora_A.grad.abs().sum() > 0, "A must receive gradients once B != 0"


def test_base_weights_frozen_and_unchanged_by_optimizer_step():
    m = LoRALinear(_base(), r=R, alpha=ALPHA)
    assert m.base_layer.weight.requires_grad is False
    assert m.base_layer.bias.requires_grad is False
    assert m.lora_A.requires_grad and m.lora_B.requires_grad

    w0 = m.base_layer.weight.detach().clone()
    b0 = m.base_layer.bias.detach().clone()
    a0 = m.lora_A.detach().clone()

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        m(torch.randn(8, D_IN)).sum().backward()
        opt.step()
    assert torch.equal(m.base_layer.weight, w0), "frozen weight changed!"
    assert torch.equal(m.base_layer.bias, b0), "frozen bias changed!"
    assert not torch.equal(m.lora_A, a0), "adapter A did not train"


def test_trainable_param_count_formula():
    m = LoRALinear(_base(), r=R, alpha=ALPHA)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable == R * (D_IN + D_OUT)
    assert m.lora_param_count == R * (D_IN + D_OUT)


# ---------------------------------------------------------------------------
# Injection into a Transformer backbone
# ---------------------------------------------------------------------------
def test_inject_wraps_only_q_and_v_projections():
    model = _tiny_bert(n_layers=3)
    adapted = apply_lora(model, r=R, alpha=ALPHA)
    assert sorted(a.split(".")[-1] for a in adapted) == ["q_lin"] * 3 + ["v_lin"] * 3
    from src.lora import LoRALinear as LL
    wrapped = [n for n, mod in model.named_modules() if isinstance(mod, LL)]
    assert sorted(wrapped) == sorted(adapted)
    # out_lin / k_lin must NOT be wrapped
    assert not any(n.endswith("out_lin") or n.endswith("k_lin") for n in wrapped)
    assert not any(n.endswith(".classifier") or "pre_classifier" in n and
                   "LoRA" in type(n).__name__ for n in wrapped)


def test_injection_is_idempotent():
    model = _tiny_bert(n_layers=2)
    first = inject_lora(model, r=R, alpha=ALPHA, freeze_backbone=True)
    second = inject_lora(model, r=R, alpha=ALPHA, freeze_backbone=True)
    assert second == [], "second injection must be a no-op"
    assert len(first) == 4


def test_injection_rejects_unknown_targets():
    model = _tiny_bert()
    with pytest.raises(ValueError):
        inject_lora(model, target_names=("does_not_exist",))


def test_frozen_backbone_trainable_head_and_adapters():
    model = _tiny_bert(n_layers=2, hidden=32)
    apply_lora(model, r=R, alpha=ALPHA)
    rep = parameter_report(model)

    assert rep["trainable_params"] == rep["lora_params"] + rep["head_params"]
    assert rep["lora_params"] == 4 * R * (32 + 32)  # 2 layers x (q,v)
    for name, p in model.named_parameters():
        leaf = name.split(".")[-1]
        is_head = name.split(".")[0] in HEAD_PREFIXES
        if leaf in ("lora_A", "lora_B") or is_head:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"


def test_tiny_model_end_to_end_step():
    """Smoke: forward/backward/step on the full model with labels+mask."""
    torch.manual_seed(0)
    model = _tiny_bert(n_layers=2)
    apply_lora(model, r=R, alpha=ALPHA)
    probe = model.get_submodule(
        "distilbert.transformer.layer.0.attention.q_lin"
    ).base_layer.weight.detach().clone()

    ids = torch.randint(0, 100, (4, 16))
    mask = torch.ones_like(ids)
    mask[:, 10:] = 0
    labels = torch.tensor([0, 1, 0, 1])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(2):
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[0] > 0  # finite, produced by the model
    cur = model.get_submodule(
        "distilbert.transformer.layer.0.attention.q_lin"
    ).base_layer.weight
    assert torch.equal(cur, probe), "backbone changed during LoRA training"

    # LoRA adapters actually received gradients at some point
    lora_grads = [
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if n.endswith("lora_B") and p.grad is not None
    ]
    assert lora_grads and max(lora_grads) > 0


def test_classifier_head_trains_for_full_ft_baseline():
    model = _tiny_bert(n_layers=1)
    for p in model.parameters():
        p.requires_grad_(True)  # full fine-tuning
    rep = parameter_report(model)
    assert rep["trainable_params"] == rep["total_params"]
    assert rep["trainable_pct"] == pytest.approx(100.0)


def test_lora_state_dict_contains_only_adapters_and_head():
    model = _tiny_bert(n_layers=1)
    apply_lora(model, r=R, alpha=ALPHA)
    from src.model import lora_state_dict
    sd = lora_state_dict(model)
    assert sd, "adapter checkpoint must not be empty"
    for k in sd:
        assert k.endswith(("lora_A", "lora_B")) or k.split(".")[0] in HEAD_PREFIXES
    n_adapter_params = sum(v.numel() for v in sd.values())
    total = sum(p.numel() for p in model.parameters())
    assert n_adapter_params < 0.2 * total, "adapter checkpoint should be small"


def test_source_does_not_use_peft():
    """The core implementation must not depend on PEFT or adapter libs."""
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    banned = ("import peft", "from peft", "get_peft_model", "lora_config", "LoraConfig")
    for fname in os.listdir(src_dir):
        if not fname.endswith(".py"):
            continue
        text = open(os.path.join(src_dir, fname)).read()
        for token in banned:
            assert token not in text, f"{fname} uses banned token {token!r}"
