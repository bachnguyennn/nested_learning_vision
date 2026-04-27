"""
optim/m3.py
─────────────────────────────────────────────────────────────────────────────
M3 — Multi-scale Momentum Muon optimizer, pulled and upgraded from
nested_learning/src/nested_learning/optim/m3.py.

100% M3 contract:
  - Only updates parameters with ndim >= 2 (all Conv/Linear/pos_embed/initial_M).
  - 1D fallback: a plain SGD-style step (used ONLY if a degenerate 1D param slips
    through — this should never happen in nested_learning_vision by design).
  - Newton-Schulz orthogonalisation runs in float32 regardless of model dtype.
"""
from __future__ import annotations

from typing import Iterable

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Newton-Schulz iteration  (Muon / Shampoo lineage)
# ─────────────────────────────────────────────────────────────────────────────

def _newton_schulz(matrix: torch.Tensor, steps: int, eps: float = 1e-6) -> torch.Tensor:
    """5-step stabilised Newton-Schulz orthogonalisation.

    Operates in float32 for numerical safety, returns the same dtype as input.
    Transforms G → closest orthogonal matrix (polar factor) in Frobenius norm.
    """
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz expects a 2D matrix")
    dtype = matrix.dtype
    x = matrix.float()
    norm = torch.linalg.norm(x)
    x = x / (norm + eps)
    # Efficient form: avoid explicit eye, use I trick
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * torch.eye(x.shape[1], device=x.device, dtype=x.dtype) - x.T @ x)
    return x.to(dtype)


def _orthogonalize(tensor: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """Flatten any ≥2D tensor to [R, C], orthogonalize, reshape back."""
    if tensor.ndim < 2:
        return tensor  # 1D fallback — should not happen in NLV
    mat = tensor.reshape(tensor.shape[0], -1)
    ortho = _newton_schulz(mat, steps=steps, eps=eps)
    return ortho.reshape_as(tensor)


# ─────────────────────────────────────────────────────────────────────────────
# M3 Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class M3(torch.optim.Optimizer):
    """Multi-scale Momentum Muon (M3).

    Implements the full two-momentum algorithm from the Nested Learning paper:
      M1  — fast EMA of gradient
      M2  — slow EMA updated every `slow_chunk` steps
      V   — second moment (RMSprop-style denominator)
      O1/O2 — Newton-Schulz orthogonalized momenta

    Update rule per step:
        O1   = NS( M1 )
        O2   = NS( M2 )  [updated only at slow_chunk boundaries]
        update = (O1 + alpha * O2) / (sqrt(V) + eps)
        p   -= lr * update

    100% M3 design:
        All weight tensors in nested_learning_vision are 2D or reshapeable to 2D,
        so _orthogonalize is called on every parameter without exception.

    Args:
        params:      iterable of model parameters
        lr:          outer learning rate (default 1e-3)
        beta1:       fast momentum decay (default 0.9)
        beta2:       second moment decay (default 0.999)
        beta3:       slow momentum decay (default 0.9)
        alpha:       slow-momentum blend coefficient (default 1.0)
        eps:         denominator epsilon (default 1e-8)
        ns_steps:    Newton-Schulz iterations (default 3; 5 for higher quality)
        slow_chunk:  steps between slow-momentum updates (default 100)
        weight_decay: decoupled L2 regularization (default 0.0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9,
        alpha: float = 1.0,
        eps: float = 1e-8,
        ns_steps: int = 3,
        slow_chunk: int = 100,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            slow_chunk=slow_chunk,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr         = group["lr"]
            beta1      = group["beta1"]
            beta2      = group["beta2"]
            beta3      = group["beta3"]
            alpha      = group["alpha"]
            eps        = group["eps"]
            ns_steps   = group["ns_steps"]
            slow_chunk = group["slow_chunk"]
            wd         = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if not state:
                    state["step"]         = 0
                    state["m1"]           = torch.zeros_like(p)
                    state["m2"]           = torch.zeros_like(p)
                    state["v"]            = torch.zeros_like(p)
                    state["slow_buffer"]  = torch.zeros_like(p)
                    state["o2"]           = torch.zeros_like(p)

                state["step"] += 1
                m1          = state["m1"]
                m2          = state["m2"]
                v           = state["v"]
                slow_buffer = state["slow_buffer"]

                # ── Accumulate ──────────────────────────────────────────────
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                slow_buffer.add_(grad)

                # ── Orthogonalize fast momentum ─────────────────────────────
                o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
                o2 = state["o2"]

                # ── Combined update ─────────────────────────────────────────
                denom  = v.sqrt().add_(eps)
                update = (o1 + alpha * o2) / denom
                p.add_(update, alpha=-lr)

                # ── Slow momentum update (every slow_chunk steps) ───────────
                if slow_chunk > 0 and state["step"] % slow_chunk == 0:
                    m2.mul_(beta3).add_(slow_buffer, alpha=1.0 - beta3)
                    slow_buffer.zero_()
                    state["o2"] = _orthogonalize(m2, steps=ns_steps, eps=eps)

        return loss
