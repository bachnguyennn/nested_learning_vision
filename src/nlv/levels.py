"""
levels.py
─────────────────────────────────────────────────────────────────────────────
Ported verbatim from nested_learning/src/nested_learning/levels.py.
Provides LevelSpec, LevelClock, and LevelState — the deterministic scheduler
for the 3-tier (Slow / Mid / Fast) nested update curriculum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence


@dataclass(frozen=True)
class LevelSpec:
    """Configuration for one nested-learning tier."""

    name: str
    update_period: int       # update every N outer steps
    warmup_steps: int = 0
    jitter: int = 0
    optimizer_key: str | None = None

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            raise ValueError(f"update_period for level {self.name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps for level {self.name} must be non-negative")
        if self.jitter < 0:
            raise ValueError(f"jitter for level {self.name} must be non-negative")


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for Nested Learning tier updates."""

    def __init__(self, specs: Sequence[LevelSpec]) -> None:
        self._specs: Dict[str, LevelSpec] = {s.name: s for s in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names provided to LevelClock")
        self._state: MutableMapping[str, LevelState] = {
            name: LevelState() for name in self._specs
        }
        self._step: int = 0

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        self._step += 1

    def should_update(self, name: str) -> bool:
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        period = spec.update_period
        if spec.jitter:
            period = period + (self._step % (spec.jitter + 1))
        return state.last_step < 0 or delta >= period

    def record_update(self, name: str) -> None:
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1

    def stats(self) -> Dict[str, LevelState]:
        return {n: LevelState(s.last_step, s.updates) for n, s in self._state.items()}


def ensure_level_specs(entries: Iterable[LevelSpec]) -> List[LevelSpec]:
    """Validate and order level specs (no duplicates)."""
    specs = list(entries)
    seen: set[str] = set()
    ordered: List[LevelSpec] = []
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"Duplicate level spec: {spec.name}")
        seen.add(spec.name)
        ordered.append(spec)
    return ordered
