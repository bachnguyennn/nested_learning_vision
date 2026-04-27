"""
hope_block.py
─────────────────────────────────────────────────────────────────────────────
Ported and unified from:
  • vit_m3_project/models.py   — ChunkedGatedDeltaNode, HOPEBlock
  • Unified_HOPE/core/hope_block.py — DeltaNode conventions

100% M3 compatible:
  - All nn.Linear weights are 2D. ✓
  - initial_M is [1, H, Dh, Dh] → 4D, flattened to 2D by M3._orthogonalize. ✓
  - RMSNorm is parameterless. ✓
  - bias=False everywhere. ✓

Delta Rule (Gated, chunked):
    Write: u_t = η(v_t − v̂_t) − β·v̂_t
    M_t   = M_{t-1} + Σ u_t ⊗ k_t   (outer product, per chunk)
    Read:  o_t = M_t q_t              (after write — paper Algorithm 1)

Sequence convention:  [B, S, D]  →  [B, S, D]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm (parameterless — 0 extra optimizer entries)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Normalization without learnable scale/shift.

    Parameterless by design: every gamma/beta 1D tensor is eliminated to
    guarantee 100% M3 (Newton-Schulz only handles ≥2D tensors).
    """
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# ChunkedGatedDeltaNode — the fast-weight memory engine
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedGatedDeltaNode(nn.Module):
    """
    Causal recurrent fast-weight memory with chunked forward-substitution.

    initial_M: [1, H, Dh, Dh] — meta-learned M₀, optimized by M3.

    Write rule:  u_t = η(v - vhat) - β·vhat      (Gated Delta)
    Read  rule:  o_t = M_t q_t                    (Algorithm 1)
    Memory:      M_{t+1} = M_t + Σ u_t ⊗ k_t
    """

    def __init__(self, num_heads: int, head_dim: int, chunk_size: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.chunk_size = chunk_size
        # Meta-learned initial memory — 4D but M3 flattens to [1*H, Dh*Dh] → 2D ✓
        self.initial_M = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))

    def reset_memory(self) -> None:
        """Clear stored memory state (call between independent sequences)."""
        self._memory_state: torch.Tensor | None = None

    def _resolve_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        if state is not None:
            return state
        cached = getattr(self, "_memory_state", None)
        if cached is not None:
            if cached.shape[0] == batch_size:
                return cached
            if cached.shape[0] > batch_size:
                return cached[:batch_size]
            pad = torch.zeros(
                batch_size - cached.shape[0], self.num_heads, self.head_dim, self.head_dim,
                device=device, dtype=dtype,
            )
            return torch.cat([cached, pad], dim=0)
        return self.initial_M.expand(batch_size, -1, -1, -1).to(dtype=dtype)

    def forward(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        beta: torch.Tensor, eta: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k, v : [B, S, H, Dh]
        beta, eta: [B, S, H, 1]
        state   : [B, H, Dh, Dh] | None
        Returns : output [B, S, H, Dh], new_M [B, H, Dh, Dh]
        """
        bsz, seq_len, nheads, hdim = k.shape
        device, dtype = k.device, k.dtype

        # QK-Norm: stabilises FP16/BF16, keeps memory norm bounded
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)

        M = self._resolve_state(bsz, device, dtype, state)

        csz = self.chunk_size
        num_chunks = (seq_len + csz - 1) // csz
        seq_pad    = num_chunks * csz
        pad_len    = seq_pad - seq_len

        # Pad to chunk multiple, zero-mask the padding
        if pad_len > 0:
            pad_kv = torch.zeros(bsz, pad_len, nheads, hdim, device=device, dtype=dtype)
            pad_g  = torch.zeros(bsz, pad_len, nheads, 1,    device=device, dtype=dtype)
            k, v, q, beta, eta = (
                torch.cat([k,    pad_kv], dim=1),
                torch.cat([v,    pad_kv], dim=1),
                torch.cat([q,    pad_kv], dim=1),
                torch.cat([beta, pad_g],  dim=1),
                torch.cat([eta,  pad_g],  dim=1),
            )
        # Zero-mask padding positions so they never corrupt memory
        valid = (torch.arange(seq_pad, device=device) < seq_len).view(1, seq_pad, 1, 1).to(dtype)
        k, v, q, beta, eta = k*valid, v*valid, q*valid, beta*valid, eta*valid

        # Reshape to chunks
        k_c    = k.view(bsz, num_chunks, csz, nheads, hdim)
        v_c    = v.view(bsz, num_chunks, csz, nheads, hdim)
        q_c    = q.view(bsz, num_chunks, csz, nheads, hdim)
        beta_c = beta.view(bsz, num_chunks, csz, nheads, 1)
        eta_c  = eta.view(bsz, num_chunks, csz, nheads, 1)

        outputs: list[torch.Tensor] = []
        eye = torch.eye(csz, device=device, dtype=dtype).view(1, 1, csz, csz)

        for n in range(num_chunks):
            k_h    = k_c[:, n].permute(0, 2, 1, 3)    # [B, H, C, Dh]
            v_h    = v_c[:, n].permute(0, 2, 1, 3)
            q_h    = q_c[:, n].permute(0, 2, 1, 3)
            beta_h = beta_c[:, n].permute(0, 2, 1, 3)  # [B, H, C, 1]
            eta_h  = eta_c[:, n].permute(0, 2, 1, 3)

            # Base read from carry-in memory
            r0  = torch.einsum("bhij,bhcj->bhci", M, k_h)
            G   = torch.tril(torch.einsum("bhid,bhjd->bhij", k_h, k_h), diagonal=-1)
            lam = (eta_h + beta_h).squeeze(-1)
            L   = eye + G * lam.unsqueeze(-2)
            rhs = r0 + torch.einsum("bhij,bhjd->bhid", G, eta_h * v_h)

            # Forward substitution (closed-form, no solve_triangular needed)
            rows: list[torch.Tensor] = []
            for row in range(csz):
                b = rhs[:, :, row:row+1, :]
                if row > 0:
                    b = b - torch.matmul(L[:, :, row:row+1, :row], torch.cat(rows, dim=2))
                rows.append(b / (L[:, :, row:row+1, row:row+1] + 1e-8))
            vhat_h = torch.cat(rows, dim=2)             # [B, H, C, Dh]

            # Gated delta write
            u_h = eta_h * (v_h - vhat_h) - beta_h * vhat_h
            M   = M + torch.einsum("bhci,bhcj->bhij", u_h, k_h)

            # Read AFTER write (paper Algorithm 1)
            output_h = torch.einsum("bhij,bhcj->bhci", M, q_h)   # [B, H, C, Dh]
            outputs.append(output_h.permute(0, 2, 1, 3))           # [B, C, H, Dh]

        self._memory_state = M
        return torch.cat(outputs, dim=1)[:, :seq_len], M


# ─────────────────────────────────────────────────────────────────────────────
# HOPEBlock — wraps DeltaNode + FFN sub-layer
# ─────────────────────────────────────────────────────────────────────────────

class HOPEBlock(nn.Module):
    """
    Single HOPE recurrent block: QKV projection → DeltaNode → FFN.

    Input/output: [B, S, D]

    100% M3:
        qkv, rates, o_proj, ff all use Linear(bias=False) → 2D weight. ✓
        RMSNorm is parameterless. ✓
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        chunk_size: int = 16,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.norm   = RMSNorm(d_model)
        self.qkv    = nn.Linear(d_model, d_model * 3, bias=False)
        # Gates: beta (write/forget) and eta (learning-rate), one scalar per head per token
        self.rates  = nn.Linear(d_model, num_heads * 2, bias=False)
        self.delta  = ChunkedGatedDeltaNode(num_heads, self.head_dim, chunk_size)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff_norm = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion, bias=False),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model, bias=False),
        )

    def reset_memory(self) -> None:
        self.delta.reset_memory()

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        residual = x
        x = self.norm(x)

        q, k, v = self.qkv(x).chunk(3, dim=-1)           # each [B, S, D]
        q = q.view(B, S, H, Dh)
        k = k.view(B, S, H, Dh)
        v = v.view(B, S, H, Dh)

        rates = torch.sigmoid(self.rates(x))              # [B, S, H*2]
        beta  = rates[:, :, :H].unsqueeze(-1)             # [B, S, H, 1]
        eta   = rates[:, :, H:].unsqueeze(-1)             # [B, S, H, 1]

        output, new_state = self.delta(q, k, v, beta, eta, state)
        output = output.reshape(B, S, D)
        output = self.o_proj(output)

        x = residual + output
        x = x + self.ff(self.ff_norm(x))
        return x, new_state
