"""Custom Low-Rank Adaptation (LoRA) implementation in plain PyTorch.

Reference
---------
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
Chen, W. "LoRA: Low-Rank Adaptation of Large Language Models."
ICLR 2022. https://arxiv.org/abs/2106.09685

Paper formulation (Eq. 3 of the paper, with the scaling factor discussed
right after it)::

    h = W0 @ x + (alpha / r) * B @ A @ x

with

    W0 : (d_out, d_in)  pretrained weight matrix, FROZEN
    B  : (d_out, r)     trainable low-rank factor, initialized to ZERO
    A  : (r, d_in)      trainable low-rank factor, initialized RANDOM (Gaussian)
    r  : rank, with r << min(d_out, d_in)
    alpha : scaling hyperparameter (the paper treats alpha as a constant
            with respect to r; this project uses the documented rule
            alpha = 2 * r, so alpha/r = 2 for every rank).

The delta matrix  dW = (alpha / r) * B @ A  has rank at most r, so the
millions of entries of dW are parameterised by only r * (d_out + d_in)
numbers. Because B starts at zero, dW = 0 at initialisation and the
adapted layer initially reproduces the pretrained layer exactly.

This module uses only standard PyTorch building blocks (nn.Linear,
tensor ops). No PEFT / adapter library is used.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
    "inject_lora",
    "lora_parameter_names",
    "mark_only_lora_and_head_trainable",
    "merged_weight",
]


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A frozen pretrained ``nn.Linear`` augmented with a low-rank adapter.

    The layer computes::

        output = base_layer(x) + (alpha / r) * B(A(dropout(x)))

    where ``base_layer.weight`` (shape ``(d_out, d_in)``) is frozen, while
    ``lora_A`` (shape ``(r, d_in)``) and ``lora_B`` (shape ``(d_out, r)``)
    are the only trainable tensors.

    Parameters
    ----------
    base_layer:
        The pretrained linear layer to wrap. Its weights are frozen
        (``requires_grad=False``); its forward computation is reused as-is,
        so bias handling and dtype follow the pretrained checkpoint.
    r:
        Rank of the adapter. Must be >= 1 and small compared with
        ``min(d_in, d_out)`` for the method to be parameter efficient.
    alpha:
        Scaling hyperparameter; the effective multiplier applied to the
        adapter output is ``alpha / r`` (paper: ``h = W0 x + (alpha/r) B A x``).
    lora_dropout:
        Dropout applied to the input of ``A`` during training only
        (0.0 disables it; the assignment's default configuration uses 0.0
        and the value is recorded in the experiment config).
    freeze_base:
        If True (default) set ``requires_grad=False`` on the base weight
        and bias so the optimizer cannot update them.

    Notes
    -----
    * ``lora_A`` is initialised with Kaiming-uniform rows (a random,
      scale-corrected initialisation, i.e. the paper's "random Gaussian"
      start up to distribution choice) so the adapter can immediately
      receive gradient signal.
    * ``lora_B`` is initialised to zero, so ``B @ A = 0`` at step 0: the
      pretrained function is preserved exactly at initialisation, and the
      adapter only learns a *change* to it during training.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"LoRALinear expects an nn.Linear base layer, got {type(base_layer)!r}"
            )
        if r < 1:
            raise ValueError(f"rank r must be >= 1, got {r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        # Multiplicative factor applied to the adapter output (paper: alpha/r).
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(float(lora_dropout))

        d_in = base_layer.in_features
        d_out = base_layer.out_features
        if self.r > min(d_in, d_out):
            raise ValueError(
                f"rank r={self.r} must satisfy r <= min(d_in={d_in}, d_out={d_out})"
            )

        # A: (r, d_in), random initialisation (Kaiming-uniform, fan_in).
        self.lora_A = nn.Parameter(torch.empty(self.r, d_in))
        # B: (d_out, r), ZERO initialisation -> zero update at start.
        self.lora_B = nn.Parameter(torch.zeros(d_out, self.r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if freeze_base:
            self.freeze_base()

    # -- helpers ----------------------------------------------------------
    def freeze_base(self) -> None:
        """Freeze the pretrained weights of the wrapped linear layer."""
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def lora_param_count(self) -> int:
        """Number of trainable adapter parameters: r * (d_in + d_out)."""
        return self.r * (self.in_features + self.out_features)

    def merged_weight(self) -> torch.Tensor:
        """Return W' = W + (alpha / r) * B @ A  (shape ``(d_out, d_in)``).

        Used for verification and for the optional merge-at-inference path:
        after merging, the module can be served as a plain ``nn.Linear``
        with no extra latency (paper, Sec. 4).
        """
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            return self.base_layer.weight + delta.to(self.base_layer.weight.dtype)

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``base(x) + (alpha/r) * B(A(dropout(x)))``.

        Shapes: ``x`` is ``(..., d_in)``; the result is ``(..., d_out)``.
        Gradients flow only through ``lora_A``/``lora_B`` (the base layer
        output is computed under ``no_grad``-free code but its parameters
        have ``requires_grad=False``, so no gradient is accumulated for them).
        """
        base_out = self.base_layer(x)              # frozen pretrained path
        # linear(linear(x, A), B): F.linear applies a raw weight matrix
        # (in_features -> out_features), so A maps (..., d_in) -> (..., r)
        # and B maps (..., r) -> (..., d_out).
        x_d = self.lora_dropout(x.to(self.lora_A.dtype))
        lora_out = F.linear(F.linear(x_d, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, "
            f"dropout={self.lora_dropout.p}"
        )


def merged_weight(module: nn.Module) -> torch.Tensor:
    """Convenience wrapper: W' of a :class:`LoRALinear` (for tests/inspection)."""
    if not isinstance(module, LoRALinear):
        raise TypeError("merged_weight expects a LoRALinear module")
    return module.merged_weight()


# ---------------------------------------------------------------------------
# Injection utilities
# ---------------------------------------------------------------------------
def inject_lora(
    model: nn.Module,
    target_names: Sequence[str] = ("q_lin", "v_lin"),
    r: int = 4,
    alpha: Optional[float] = None,
    lora_dropout: float = 0.0,
    freeze_backbone: bool = True,
) -> List[str]:
    """Replace matching ``nn.Linear`` submodules with :class:`LoRALinear`.

    Target modules are identified by their *leaf name* so the same call works
    for DistilBERT (``...attention.q_lin`` / ``...attention.v_lin``) and for
    other Transformer implementations that name their query/value projections
    ``q_proj``/``v_proj``.

    Parameters
    ----------
    model:
        Model whose attention projections will be adapted, modified in place.
    target_names:
        Leaf module names to adapt. Default: query and value projections
        (``q_lin``, ``v_lin``), following the paper, which adapts ``W_q`` and
        ``W_v`` in most experiments.
    r:
        Adapter rank.
    alpha:
        Scaling hyperparameter. Defaults to ``2 * r`` (documented project
        rule: ``alpha = 2r``, hence ``alpha / r = 2`` for all ranks).
    lora_dropout:
        Dropout on the adapter input (0.0 by default; recorded in configs).
    freeze_backbone:
        If True, freeze every parameter in the model first (callers then
        re-enable the classification head explicitly).

    Returns
    -------
    list of str
        Fully-qualified names of the modules that were wrapped.

    Raises
    ------
    ValueError
        If no module matched ``target_names`` (guards against silent typos
        in target names producing a "LoRA" run that trains nothing).
    """
    if alpha is None:
        alpha = 2.0 * r  # project rule: alpha = 2 * r

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)

    injected: List[str] = []
    # Collect first: mutating while iterating named_modules() is unsafe.
    candidates: List[Tuple[str, nn.Module]] = [
        (name, mod)
        for name, mod in model.named_modules()
        if name.split(".")[-1] in target_names
    ]

    already_wrapped = 0
    for name, module in candidates:
        if isinstance(module, LoRALinear):
            already_wrapped += 1  # idempotent: never wrap a wrapper twice
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr,
            LoRALinear(module, r=r, alpha=alpha, lora_dropout=lora_dropout),
        )
        injected.append(name)

    if not injected and already_wrapped == 0:
        raise ValueError(
            f"inject_lora found no nn.Linear modules named {tuple(target_names)}; "
            "check target_names against the model architecture."
        )
    return injected


def lora_parameter_names(model: nn.Module) -> List[str]:
    """Return fully-qualified names of all adapter parameters (A and B)."""
    return [
        n for n, _ in model.named_parameters()
        if n.endswith("lora_A") or n.endswith("lora_B")
    ]


def mark_only_lora_and_head_trainable(
    model: nn.Module,
    head_prefixes: Iterable[str] = ("classifier", "pre_classifier"),
) -> List[str]:
    """Freeze everything, then re-enable LoRA adapters + classification head.

    The pretrained backbone stays frozen; only ``lora_A``/``lora_B`` and the
    head parameters whose qualified name starts with one of ``head_prefixes``
    become trainable. Returns the list of trainable parameter names.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    trainable: List[str] = []
    for name, p in model.named_parameters():
        leaf = name.split(".")[-1]
        if leaf in ("lora_A", "lora_B") or any(
            name.split(".")[0] == pref or name.startswith(pref + ".")
            for pref in head_prefixes
        ):
            p.requires_grad_(True)
            trainable.append(name)
    return trainable
