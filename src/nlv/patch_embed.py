"""
patch_embed.py
─────────────────────────────────────────────────────────────────────────────
Vision-specific input tokenizers for nested_learning_vision.

All layers use bias=False — ensuring 100% M3 compatibility (no 1D parameters).

Provided tokenizers:
  FlatPatchEmbed   — single Conv2d (fast, CIFAR-scale)
  ConvStemEmbed    — 3-layer conv stem (stable, ImageNet-scale)

The CLS "token" is NOT a learnable nn.Parameter in this project.
Instead, a zero-content slot is appended to the patch sequence and receives
its semantic meaning purely from the learned positional embedding. This
preserves 100% M3 since pos_embed [1, N+1, D] is 3D → reshaped to [N+1, D]
during the optimizer step. See optim/m3.py _orthogonalize.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlatPatchEmbed(nn.Module):
    """
    Single-convolution patch tokenizer.

    image [B, C, H, W]  →  tokens [B, num_patches, d_model]

    M3 compatibility:
        Conv2d weight is [d_model, in_channels, P, P] — 4D, flattened to
        [d_model, in_channels*P*P] by M3._orthogonalize. ✓
        bias=False — no 1D parameters. ✓
    """

    def __init__(
        self,
        d_model: int = 192,
        patch_size: int = 4,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_model    = d_model
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, N, d_model] where N = (H/P)*(W/P)."""
        x = self.proj(x)           # [B, D, H/P, W/P]
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1, 2)   # [B, N, D]


class ConvStemEmbed(nn.Module):
    """
    3-layer convolutional stem followed by a final strided projection.
    More stable than FlatPatchEmbed for deeper models (ImageNet-scale).

    M3 compatibility:
        All Conv2d weights are 4D. bias=False everywhere. ✓
    """

    def __init__(
        self,
        d_model: int = 384,
        patch_size: int = 16,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        mid = d_model // 2
        # 3-stage stem: progressively increase channels, reduce spatial res
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels,    mid // 2, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid // 2,  mid,     3, stride=2, padding=1, bias=False),
            nn.GELU(),
        )
        # Final patch projection (large kernel → non-overlapping patches)
        remaining_stride = patch_size // 4   # after 2×2 downsampling in stem
        self.proj = nn.Conv2d(mid, d_model, remaining_stride, stride=remaining_stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # [B, mid, H/4, W/4]
        x = self.proj(x)     # [B, D,   H/P, W/P]
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1, 2)   # [B, N, D]
