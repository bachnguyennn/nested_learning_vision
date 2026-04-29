# Changelog (audit fixes — 2026-04-29)

A file-by-file summary of the changes made to address the audit findings.

## `train_cifar100.py`

- **EWC reworked.** The old `EWC.snapshot()` captured Fisher information
  from whatever single gradient happened to be present and was called
  every slow update inside the training loop, producing a moving anchor
  that is not classical EWC. The new `EWC.snapshot(loader, device,
  criterion, n_batches)` runs a small loop over Task-A data, averages
  squared gradients to estimate the diagonal Fisher, and saves a single
  weight anchor at that moment. Penalty: `λ · Σ F_i (θ_i − θ_i*)²`.
- **`GradScaler` removed.** Training uses `torch.bfloat16` autocast on CUDA;
  bf16 has no underflow regime that `GradScaler` is designed to fix, so
  using it together with the per-tier optimizer step path was both
  unnecessary and fragile. We now do plain `loss.backward()` and let each
  tier optimizer step directly.
- **`ran` is always defined.** Previously, on a step with NaN gradients
  the variable was never bound, which would raise `NameError` if the
  first step ever produced non-finite gradients. We now initialise
  `ran: dict[str, bool] = {}` ahead of the finite-grad branch.
- **EWC instantiation note** clarifies that single-task training never
  calls `snapshot`, so `ewc.penalty()` returns 0 — the regularizer is
  inert until anchored externally.

## `src/nlv/model.py`

- **`VisionModelConfig.reset_memory_per_forward`** (`bool`, default
  `True`) now controls whether the recurrent fast-weight memory inside
  each `HOPEBlock` is reset at the start of `forward`. Default behaviour
  is unchanged (i.i.d. classification); set `False` for streaming /
  TBPTT use cases.
- **Unused tier-parameter helpers removed.** `slow_parameters /
  mid_parameters / fast_parameters` on the model were never read by the
  optimizer manager and could drift from the manager’s own routing.
  Tier→param routing is now owned exclusively by
  `TieredOptimizerManager._get_tier_params`.

## `src/nlv/optim/manager.py`

- **Cross-tier overlap guard.** Construction now tracks parameter
  identity globally and raises `ValueError` if a parameter is routed to
  more than one tier. Prevents silent double-stepping if the routing
  table is ever expanded.

## `src/nlv/patch_embed.py`

- Dropped unused `B, D, h, w = x.shape` unpacks in
  `FlatPatchEmbed.forward` and `ConvStemEmbed.forward`.

## `tests/` (new)

- `tests/conftest.py` — adds `src/` to `sys.path` so tests can import
  `nlv` without installing the package.
- `tests/test_levels.py` — `LevelClock` cadence, warmup, jitter,
  duplicate detection, ordering helpers.
- `tests/test_m3.py` — Newton-Schulz finiteness/bounded singular values,
  4-D tensor reshape correctness, `slow_chunk` cadence on `o2`,
  graceful no-grad behaviour.
- `tests/test_delta_node.py` — `RMSNorm` / `ChunkedGatedDeltaNode` /
  `HOPEBlock` shape, padding correctness, gradient flow.
- `tests/test_manager.py` — default routing has no cross-tier overlap;
  cadence fires the right tiers; cross-tier overlap guard raises.
- `tests/test_cf_benchmark.py` — synthetic-data integration test for the
  CF benchmark across all four variants.

## `cf_benchmark.py` (new)

A self-contained, ablation-aware catastrophic-forgetting harness; see
[`cf_benchmark.md`](cf_benchmark.md) for the full design.

## `docs/` (new)

- `README.md`, `audit_2026-04-29.md`, `changelog.md`, `testing.md`,
  `cf_benchmark.md`, `architecture.md`.
