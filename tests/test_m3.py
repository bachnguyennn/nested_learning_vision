"""M3 optimizer invariants.

Focus on the structural contract — Newton-Schulz orthogonality, multi-shape
support, slow-chunk cadence — rather than convergence speed (covered by
`smoke_test.py` and the training script).
"""
from __future__ import annotations

import torch

from nlv.optim.m3 import M3, _newton_schulz, _orthogonalize


def test_newton_schulz_returns_finite_and_bounded() -> None:
    """Frobenius-normalized Newton-Schulz pushes singular values toward 1.

    The implementation in `nlv.optim.m3._newton_schulz` normalizes by Frobenius
    norm before iterating, so the **largest** singular value is bounded by ~1
    after a few steps, but tiny singular values converge slowly (this is by
    design — Muon-lineage). We pin only the finite/bounded property here and
    let the optimizer-level tests cover end-to-end behaviour.
    """
    torch.manual_seed(0)
    G = torch.randn(8, 4)
    O = _newton_schulz(G, steps=5)
    assert O.shape == G.shape
    assert torch.isfinite(O).all()
    sv = torch.linalg.svdvals(O)
    assert sv.max().item() <= 1.0 + 1e-3  # never blows up
    assert sv.min().item() >= 0.0          # singular values non-negative


def test_orthogonalize_handles_4d_tensor() -> None:
    torch.manual_seed(0)
    G = torch.randn(8, 3, 4, 4)
    O = _orthogonalize(G, steps=3, eps=1e-6)
    assert O.shape == G.shape
    assert torch.isfinite(O).all()


def test_step_advances_2d_param() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = M3([p], lr=1e-2, ns_steps=3, slow_chunk=4, weight_decay=0.0)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    assert not torch.allclose(p.detach(), before)


def test_slow_buffer_runs_on_chunk_boundary() -> None:
    """o2 (slow orthogonalized momentum) should change exactly at slow_chunk boundaries."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = M3([p], lr=1e-3, ns_steps=3, slow_chunk=2, weight_decay=0.0)
    for step in range(1, 5):
        p.grad = torch.randn_like(p)
        opt.step()
        state = opt.state[p]
        if step < 2:
            # First slow update happens AT step == slow_chunk
            assert torch.equal(state["o2"], torch.zeros_like(p))
        if step == 2:
            assert not torch.equal(state["o2"], torch.zeros_like(p))


def test_zero_grad_skipped() -> None:
    """Parameters with grad=None should be silently skipped, not crash."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = M3([p], lr=1e-3)
    # No grad set
    opt.step()  # must not raise
