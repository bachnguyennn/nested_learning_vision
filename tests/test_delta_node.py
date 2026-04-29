"""ChunkedGatedDeltaNode + HOPEBlock invariants.

These tests intentionally avoid asserting *exact* numerical values so they
remain stable across PyTorch versions; instead they pin shape, finiteness,
chunk-padding correctness, and gradient flow.
"""
from __future__ import annotations

import torch

from nlv.hope_block import ChunkedGatedDeltaNode, HOPEBlock, RMSNorm


def test_rmsnorm_shape_and_finite() -> None:
    n = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    y = n(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_delta_node_shape_full_chunk() -> None:
    H, Dh, C, B, S = 4, 8, 4, 2, 8
    node = ChunkedGatedDeltaNode(H, Dh, chunk_size=C)
    q = torch.randn(B, S, H, Dh)
    k = torch.randn(B, S, H, Dh)
    v = torch.randn(B, S, H, Dh)
    beta = torch.rand(B, S, H, 1)
    eta = torch.rand(B, S, H, 1)
    out, M = node(q, k, v, beta, eta)
    assert out.shape == (B, S, H, Dh)
    assert M.shape == (B, H, Dh, Dh)
    assert torch.isfinite(out).all()
    assert torch.isfinite(M).all()


def test_delta_node_shape_with_padding() -> None:
    """S not multiple of chunk_size should still produce length-S output."""
    H, Dh, C, B, S = 2, 4, 4, 1, 7  # 7 = 1 chunk of 4 + 1 partial chunk of 3
    node = ChunkedGatedDeltaNode(H, Dh, chunk_size=C)
    q, k, v = (torch.randn(B, S, H, Dh) for _ in range(3))
    beta = torch.rand(B, S, H, 1)
    eta = torch.rand(B, S, H, 1)
    out, _ = node(q, k, v, beta, eta)
    assert out.shape == (B, S, H, Dh)


def test_hope_block_grad_flow() -> None:
    blk = HOPEBlock(d_model=32, num_heads=4, chunk_size=4)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y, _ = blk(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in blk.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_hope_block_residual_path() -> None:
    """At init, FF is small but not zero; output should differ from input."""
    torch.manual_seed(0)
    blk = HOPEBlock(d_model=16, num_heads=4, chunk_size=4)
    x = torch.randn(1, 4, 16)
    y, _ = blk(x)
    assert y.shape == x.shape
    assert not torch.equal(y, x)
