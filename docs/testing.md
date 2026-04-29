# Testing

There are now two layers of automated checks:

1. `smoke_test.py` — single-file end-to-end sanity (zero 1-D params,
   forward, backward, `TieredOptimizerManager.step`, checkpoint
   roundtrip). It is the same script that previously documented the
   project’s “100% M3” claim and still passes.
2. `tests/` — focused unit / integration tests using `pytest`.

## Run

```bash
# Single-file smoke
python smoke_test.py

# Full test suite (no dataset download)
python -m pytest tests/ -q

# Verbose
python -m pytest tests/ -v
```

## What is covered

| File                          | What it pins                                                                                  |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `tests/test_levels.py`        | Cadence, warmup, jitter, duplicate detection in `LevelClock` / `LevelSpec`.                    |
| `tests/test_m3.py`            | Newton-Schulz finite & singular-value-bounded, 4-D tensor reshape, `slow_chunk` cadence.       |
| `tests/test_delta_node.py`    | Shape and gradient flow for `RMSNorm`, `ChunkedGatedDeltaNode`, `HOPEBlock`, padding correctness. |
| `tests/test_manager.py`       | Default tier routing has no cross-tier overlap; correct tiers fire on a known schedule; cross-tier overlap guard raises `ValueError`. |
| `tests/test_cf_benchmark.py`  | Synthetic-data integration test that runs **all four** ablation variants of `cf_benchmark.run_variant` end-to-end. |

## Why these pin shape/finite, not numerics

The numeric path of `ChunkedGatedDeltaNode` is hand-derived; the safest
contract to enforce in CI is *shape* + *finiteness* + *gradient flow*.
End-to-end accuracy is covered by `train_cifar100.py` and
`cf_benchmark.py`. We do **not** assert exact numbers in the tests so
they remain stable across PyTorch versions.

## Adding a new test

- Place new files under `tests/` and start their names with `test_`.
- They get `src/` on `sys.path` automatically via `tests/conftest.py`.
- Prefer **shape + finite** assertions; avoid pinning floats unless you
  also pin the seed and PyTorch version.
- If the test needs a model, `VisionModelConfig(d_model=32, num_heads=4,
  num_slow=1, num_mid=1, num_fast=1)` is fast enough for CI.
