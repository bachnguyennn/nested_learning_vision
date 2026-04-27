"""
__init__.py — public API for the nlv package.
"""
from .model       import NestedVisionModel, VisionModelConfig
from .hope_block  import HOPEBlock, ChunkedGatedDeltaNode, RMSNorm
from .patch_embed import FlatPatchEmbed, ConvStemEmbed
from .levels      import LevelSpec, LevelClock
from .device      import resolve_device, auto_device
from .optim       import M3, TieredOptimizerManager, TierConfig

__all__ = [
    "NestedVisionModel",
    "VisionModelConfig",
    "HOPEBlock",
    "ChunkedGatedDeltaNode",
    "RMSNorm",
    "FlatPatchEmbed",
    "ConvStemEmbed",
    "LevelSpec",
    "LevelClock",
    "resolve_device",
    "auto_device",
    "M3",
    "TieredOptimizerManager",
    "TierConfig",
]
