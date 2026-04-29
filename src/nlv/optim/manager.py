"""
optim/manager.py
─────────────────────────────────────────────────────────────────────────────
TieredOptimizerManager — wires three M3 instances (Slow / Mid / Fast) to
the 3-tier nested vision model, using LevelClock for deterministic scheduling.

Derived from nested_learning/src/nested_learning/optim/manager.py but
simplified: we remove the DeepMomentum fallback entirely. Every parameter
group is handled by M3. Period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from ..levels import LevelClock, LevelSpec
from .m3 import M3


@dataclass
class TierConfig:
    """Configuration bundle for one optimizer tier."""
    name: str                           # "slow" | "mid" | "fast"
    update_period: int                  # step cadence (e.g. 256 / 16 / 1)
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.9
    alpha: float = 1.0
    ns_steps: int = 3
    slow_chunk: int = 100
    weight_decay: float = 0.0
    warmup_steps: int = 0


class TieredOptimizerManager:
    """
    Manages three M3 optimizers (Slow / Mid / Fast) with the LevelClock
    scheduler driving their update cadences.

    Usage in training loop:
        mgr = TieredOptimizerManager(model, [slow_cfg, mid_cfg, fast_cfg])
        ...
        loss.backward()
        mgr.step()          # applies all tiers that are due this step
        mgr.zero_grad()
        mgr.tick()          # advance the clock
    """

    def __init__(self, model: nn.Module, tier_configs: Sequence[TierConfig]) -> None:
        self.model = model
        self.tier_configs: Dict[str, TierConfig] = {t.name: t for t in tier_configs}

        # Build LevelSpecs for the clock
        specs = [
            LevelSpec(name=t.name, update_period=t.update_period, warmup_steps=t.warmup_steps)
            for t in tier_configs
        ]
        self.clock = LevelClock(specs)

        # Build one M3 per tier, each wrapping the correct parameter group.
        # Cross-tier overlap is forbidden: a parameter must belong to exactly
        # ONE optimizer to avoid double-stepping.
        self.optimizers: Dict[str, M3] = {}
        global_seen: Dict[int, str] = {}
        for t in tier_configs:
            params = self._get_tier_params(t.name)
            if not params:
                raise ValueError(f"Tier '{t.name}' has no parameters in the model.")
            for p in params:
                if id(p) in global_seen:
                    raise ValueError(
                        f"Parameter id={id(p)} is routed to both tiers "
                        f"'{global_seen[id(p)]}' and '{t.name}'. "
                        "Each parameter must belong to exactly one tier."
                    )
                global_seen[id(p)] = t.name
            self.optimizers[t.name] = M3(
                params,
                lr=t.lr,
                beta1=t.beta1,
                beta2=t.beta2,
                beta3=t.beta3,
                alpha=t.alpha,
                eps=1e-8,
                ns_steps=t.ns_steps,
                slow_chunk=t.slow_chunk,
                weight_decay=t.weight_decay,
            )

    # ── Parameter routing ─────────────────────────────────────────────────────

    def _get_tier_params(self, tier: str) -> List[torch.nn.Parameter]:
        """
        Route parameters to the correct tier.
        Convention matches nested_vision.NestedVisionModel attribute names:
            slow_layers → Slow tier
            mid_layers  → Mid tier
            fast_layers → Fast tier
            patch_embed, pos_embed, cls_slot → Fast tier (updated every step)
            norm, head   → Fast tier
        """
        m = self.model
        mapping: Dict[str, List[str]] = {
            "slow": ["slow_layers"],
            "mid":  ["mid_layers"],
            "fast": ["fast_layers", "patch_embed", "pos_embed", "norm", "head"],
        }
        attr_names = mapping.get(tier, [])
        params: List[torch.nn.Parameter] = []
        seen_ids: set[int] = set()
        for attr in attr_names:
            obj = getattr(m, attr, None)
            if obj is None:
                continue
            if isinstance(obj, nn.Parameter):
                if id(obj) not in seen_ids:
                    params.append(obj)
                    seen_ids.add(id(obj))
            elif isinstance(obj, nn.Module):
                for p in obj.parameters():
                    if p.requires_grad and id(p) not in seen_ids:
                        params.append(p)
                        seen_ids.add(id(p))
        return params

    # ── Training loop API ────────────────────────────────────────────────────

    def step(self) -> Dict[str, bool]:
        """Step all tiers whose clock says they are due. Returns which ran."""
        ran: Dict[str, bool] = {}
        for name, opt in self.optimizers.items():
            if self.clock.should_update(name):
                opt.step()
                self.clock.record_update(name)
                ran[name] = True
            else:
                ran[name] = False
        return ran

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for ALL parameter groups (call after step)."""
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def tick(self) -> None:
        """Advance the LevelClock by one outer step."""
        self.clock.tick()

    def state_dict(self) -> Dict[str, object]:
        return {
            "clock_step": self.clock.step,
            "clock_stats": {n: (s.last_step, s.updates) for n, s in self.clock.stats().items()},
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.clock._step = int(state["clock_step"])  # type: ignore[arg-type]
        for name, (last_step, updates) in state["clock_stats"].items():  # type: ignore[union-attr]
            if name in self.clock._state:
                self.clock._state[name].last_step = last_step
                self.clock._state[name].updates   = updates
        for name, opt_state in state["optimizers"].items():  # type: ignore[union-attr]
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)
