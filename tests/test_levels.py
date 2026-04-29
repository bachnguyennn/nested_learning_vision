"""LevelClock invariants: cadence, warmup, jitter, deterministic ordering."""
from __future__ import annotations

import pytest

from nlv.levels import LevelClock, LevelSpec, ensure_level_specs


def test_period_only_two_levels() -> None:
    clock = LevelClock([
        LevelSpec("fast", update_period=1),
        LevelSpec("slow", update_period=4),
    ])

    fired = {"fast": 0, "slow": 0}
    for _ in range(8):
        clock.tick()
        for name in ("fast", "slow"):
            if clock.should_update(name):
                clock.record_update(name)
                fired[name] += 1

    assert fired["fast"] == 8
    assert fired["slow"] == 2  # steps 4 and 8


def test_warmup_blocks_updates() -> None:
    """warmup_steps=3 means the clock skips updates while step < 3.

    After tick() the step is 1, 2, 3, ...; the strict-less comparison in
    `LevelClock.should_update` therefore blocks ticks 1 and 2 only.
    """
    clock = LevelClock([LevelSpec("slow", update_period=1, warmup_steps=3)])
    seen = []
    for _ in range(6):
        clock.tick()
        seen.append(clock.should_update("slow"))
        if clock.should_update("slow"):
            clock.record_update("slow")
    assert seen[:2] == [False, False]
    assert all(seen[2:])


def test_duplicate_specs_rejected() -> None:
    with pytest.raises(ValueError):
        LevelClock([LevelSpec("a", update_period=1), LevelSpec("a", update_period=2)])


def test_invalid_period_rejected() -> None:
    with pytest.raises(ValueError):
        LevelSpec("x", update_period=0)
    with pytest.raises(ValueError):
        LevelSpec("x", update_period=1, warmup_steps=-1)


def test_ensure_level_specs_orders_and_dedupes() -> None:
    specs = [LevelSpec("a", 1), LevelSpec("b", 2)]
    out = ensure_level_specs(specs)
    assert [s.name for s in out] == ["a", "b"]

    with pytest.raises(ValueError):
        ensure_level_specs([LevelSpec("a", 1), LevelSpec("a", 2)])
