"""TieredOptimizerManager: tier scheduling + cross-tier overlap guard."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nlv.model import NestedVisionModel, VisionModelConfig
from nlv.optim.manager import TierConfig, TieredOptimizerManager


def _build_model() -> NestedVisionModel:
    cfg = VisionModelConfig(
        num_classes=10, d_model=64, num_heads=4,
        num_slow=1, num_mid=1, num_fast=1,
        patch_size=4, img_size=32,
    )
    return NestedVisionModel(cfg)


def test_default_tier_routing_no_overlap() -> None:
    model = _build_model()
    mgr = TieredOptimizerManager(model, [
        TierConfig("slow", update_period=8, lr=1e-4),
        TierConfig("mid",  update_period=4, lr=5e-4),
        TierConfig("fast", update_period=1, lr=1e-3),
    ])
    seen: dict[int, str] = {}
    for tier, opt in mgr.optimizers.items():
        for grp in opt.param_groups:
            for p in grp["params"]:
                assert id(p) not in seen, (
                    f"param routed to multiple tiers: {seen[id(p)]} and {tier}"
                )
                seen[id(p)] = tier


def test_cadence_runs_correct_tiers() -> None:
    model = _build_model()
    mgr = TieredOptimizerManager(model, [
        TierConfig("slow", update_period=8, lr=1e-4),
        TierConfig("mid",  update_period=4, lr=5e-4),
        TierConfig("fast", update_period=1, lr=1e-3),
    ])
    # Smallest test: just verify which tiers are due at known steps.
    # The clock fires on step==1, then on multiples thereafter.
    schedule = []
    for _ in range(10):
        mgr.tick()
        schedule.append({n: mgr.clock.should_update(n) for n in ("fast", "mid", "slow")})
        for n in ("fast", "mid", "slow"):
            if mgr.clock.should_update(n):
                mgr.clock.record_update(n)
    # fast should be due on every tick
    assert all(s["fast"] for s in schedule)
    # slow should be due on the very first tick (initial), then again after 8
    assert schedule[0]["slow"] is True
    # ticks 1-7 (zero-indexed 1..7) should NOT fire slow
    for s in schedule[1:8]:
        assert s["slow"] is False


def test_cross_tier_overlap_rejected(monkeypatch) -> None:
    """If a parameter is routed to two tiers, manager construction must fail."""
    model = _build_model()

    # Force 'slow' and 'mid' to both claim 'fast_layers'
    def fake_get_tier_params(self, tier: str):  # noqa: ARG001
        if tier in ("slow", "mid"):
            return [next(self.model.fast_layers.parameters())]
        return list(self.model.fast_layers.parameters())[:1]

    monkeypatch.setattr(
        TieredOptimizerManager, "_get_tier_params", fake_get_tier_params, raising=True,
    )
    with pytest.raises(ValueError, match="must belong to exactly one tier"):
        TieredOptimizerManager(model, [
            TierConfig("slow", update_period=8, lr=1e-4),
            TierConfig("mid",  update_period=4, lr=5e-4),
            TierConfig("fast", update_period=1, lr=1e-3),
        ])
