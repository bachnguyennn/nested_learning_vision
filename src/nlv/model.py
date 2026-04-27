"""
model.py
─────────────────────────────────────────────────────────────────────────────
NestedVisionModel — the full 3-tier nested learning vision model.

Architecture:
    PatchEmbed  → pos_embed + CLS-slot appended at end
    Slow Tier   (slow_layers: N HOPEBlocks) — global scene structure
    Mid  Tier   (mid_layers:  N HOPEBlocks) — region-level semantics
    Fast Tier   (fast_layers: N HOPEBlocks) — fine-grained textures/edges
    RMSNorm → read CLS token → Linear head

100% M3 compatibility:
    ─ No nn.Embedding (replaced by PatchEmbed, a Conv2d)
    ─ No cls_token nn.Parameter — CLS is a zero-content slot + pos_embed[-1]
    ─ No LayerNorm (uses parameterless RMSNorm)
    ─ bias=False on all Conv2d and Linear layers
    ─ pos_embed is [1, N+1, D] → M3 flattens to [N+1, D] ✓
    ─ initial_M per HOPEBlock is [1, H, Dh, Dh] → M3 flattens to 2D ✓

Spatial-frequency tier mapping:
    Slow ← global features (updated every 256 steps, EWC-protected)
    Mid  ← object-part features (updated every 16 steps)
    Fast ← per-patch texture/edge (updated every step)

CLS placement: position N (end of sequence) — so it reads the full patch
memory via causal recurrence before classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .hope_block import HOPEBlock, RMSNorm
from .patch_embed import ConvStemEmbed, FlatPatchEmbed


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionModelConfig:
    # Task
    num_classes:   int   = 100       # classification output size

    # Architecture
    d_model:       int   = 192       # embedding dimension
    num_heads:     int   = 6         # HOPE heads (head_dim = d_model / num_heads)
    chunk_size:    int   = 16        # DeltaNode chunk size
    ff_expansion:  int   = 4         # FFN width multiplier

    # Tier depths (number of HOPEBlocks per tier)
    num_slow:      int   = 2
    num_mid:       int   = 2
    num_fast:      int   = 2

    # Input
    patch_size:    int   = 4         # patch grid stride
    in_channels:   int   = 3
    img_size:      int   = 32
    embed_type:    str   = "flat"    # "flat" | "stem"

    # Gradient checkpointing (saves memory at cost of re-compute)
    gradient_checkpointing: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class NestedVisionModel(nn.Module):
    """
    3-Tier Nested Vision Transformer with 100% M3 optimizer compatibility.

    Tier update cadences (controlled externally by TieredOptimizerManager):
        Fast  — every step       (texture / edge learning)
        Mid   — every 16 steps   (object-part learning)
        Slow  — every 256 steps  (global scene learning, EWC-protected)
    """

    def __init__(self, config: VisionModelConfig) -> None:
        super().__init__()
        self.config = config
        D = config.d_model

        # ── Patch tokenizer ────────────────────────────────────────────────
        if config.embed_type == "stem":
            self.patch_embed: nn.Module = ConvStemEmbed(D, config.patch_size, config.in_channels)
        else:
            self.patch_embed = FlatPatchEmbed(D, config.patch_size, config.in_channels)

        num_patches = (config.img_size // config.patch_size) ** 2
        seq_len     = num_patches + 1   # +1 for CLS slot at the end

        # ── Positional embedding (CLS absorbed as position N) ─────────────
        # Shape [1, seq_len, D] — M3 flattens to [seq_len, D] during step ✓
        # pos_embed[-1] IS the CLS positional encoding — no separate parameter.
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, D))

        # ── 3-Tier HOPE layers ─────────────────────────────────────────────
        def make_tier(n: int) -> nn.ModuleList:
            return nn.ModuleList([
                HOPEBlock(D, config.num_heads, config.chunk_size, config.ff_expansion)
                for _ in range(n)
            ])

        self.slow_layers = make_tier(config.num_slow)
        self.mid_layers  = make_tier(config.num_mid)
        self.fast_layers = make_tier(config.num_fast)

        # ── Output head (parameterless norm + linear) ──────────────────────
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, config.num_classes, bias=False)

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    # ── Memory management ──────────────────────────────────────────────────

    def reset_memory(self) -> None:
        """Reset all tier memories — call before each independent image batch."""
        for layer in (*self.slow_layers, *self.mid_layers, *self.fast_layers):
            layer.reset_memory()

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, H, W]  — raw image batch
        Returns: logits [B, num_classes]
        """
        B = x.shape[0]
        D = self.config.d_model

        # 1. Patch embedding: image → patch tokens [B, N, D]
        x = self.patch_embed(x)

        # 2. Append CLS slot (zero-content — position carries meaning via pos_embed)
        cls_slot = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, cls_slot], dim=1)   # [B, N+1, D]

        # 3. Add positional embeddings (including learned CLS position)
        x = x + self.pos_embed                 # [B, N+1, D]

        # 4. Reset fast-weight memories (each image is independent)
        self.reset_memory()

        # 5. Slow tier — global structure
        for layer in self.slow_layers:
            x, _ = layer(x, state=None)

        # 6. Mid tier — region semantics
        for layer in self.mid_layers:
            x, _ = layer(x, state=None)

        # 7. Fast tier — fine-grained details
        for layer in self.fast_layers:
            x, _ = layer(x, state=None)

        # 8. Read CLS token (last position — has read all patch memory)
        cls_out = self.norm(x[:, -1, :])       # [B, D]

        # 9. Classification
        return self.head(cls_out)              # [B, num_classes]

    # ── Tier parameter helpers (used by TieredOptimizerManager) ──────────

    def slow_parameters(self):
        return self.slow_layers.parameters()

    def mid_parameters(self):
        return self.mid_layers.parameters()

    def fast_parameters(self):
        """Fast tier + input/output layers (updated every step)."""
        for p in self.fast_layers.parameters():
            yield p
        yield self.pos_embed
        yield from self.patch_embed.parameters()
        yield from self.head.parameters()
