# Catastrophic Forgetting Benchmark — `cf_benchmark.py`

A self-contained continual-learning harness designed to **isolate** the
contribution of the nested-tier scheduler vs EWC vs the combination — so
results actually answer the right question.

## Why a new harness

The Kaggle scripts (`kaggle_cf_test.py`, `kaggle_cf_svhn.py`) compare a
single configuration (3-tier + EWC) against a single baseline (no
EWC). That tells you EWC helps; it does **not** isolate whether the
nested-tier scheduling itself reduces forgetting. The new harness runs
four variants on the same backbone init:

| Variant       | Tier scheduler | EWC on slow tier |
|---------------|----------------|------------------|
| `plain`       | no             | no               |
| `tiers_only`  | yes            | no               |
| `ewc_only`    | no             | yes              |
| `tiers_ewc`   | yes            | yes              |

…and on **two task pairs**:

| Task              | Sequence                       | What it stresses                |
|-------------------|--------------------------------|---------------------------------|
| `split_cifar10`   | CIFAR-10 0–4 → CIFAR-10 5–9    | same domain, new concepts       |
| `cifar10_to_svhn` | CIFAR-10 → SVHN                | cross-domain shift              |

## Metrics reported

For each variant, after training Phase B:

- **`acc_a_before`** — Task-A accuracy at the end of Phase A.
- **`acc_a_after`**  — Task-A accuracy *after* Phase B finishes.
- **`acc_b_after`**  — Task-B accuracy at the end of Phase B.
- **`forgetting`**   — `acc_a_before − acc_a_after` (lower is better).
- **`average`**      — `(acc_a_after + acc_b_after) / 2`
  (overall continual-learning quality; rewards both retention *and*
  plasticity, so a model that “preserves” Task A by failing Task B
  cannot win).
- **`wallclock_sec`** — variant runtime.

## EWC details (Kirkpatrick-style, anchored once)

The benchmark uses classical EWC:

```text
penalty = λ · Σ_i  F_i · (θ_i - θ_i*)²
```

- The diagonal Fisher `F` is estimated by averaging squared gradients
  across `--n-fisher` minibatches drawn from Phase-A data, **once**,
  immediately after Phase A.
- The anchor `θ_i*` is the slow-tier weight snapshot at that same
  moment.
- Only the slow tier is anchored — fast/mid tiers stay free to adapt
  to Task B.
- During single-task training (no Phase A→B transition) `snapshot` is
  never called, so `penalty()` returns `0`.

## Running

```bash
# Single variant
python cf_benchmark.py --task split_cifar10 --variant tiers_ewc \
    --epochs-a 5 --epochs-b 5 --device auto

# Full ablation matrix → JSON report
python cf_benchmark.py --task split_cifar10 --variant all \
    --epochs-a 5 --epochs-b 5 --output reports/cf_split.json

python cf_benchmark.py --task cifar10_to_svhn --variant all \
    --epochs-a 5 --epochs-b 5 --output reports/cf_cross.json
```

The harness:

- builds `ContinualVisionModel` (the same HOPE backbone used in
  `train_cifar100.py`, plus per-task linear heads),
- seeds the backbone identically across variants for the same task
  (`--seed`),
- uses bf16 autocast on CUDA (no `GradScaler`),
- gracefully skips optimizer steps on non-finite grads.

## Reading the output

A report block looks like:

```text
[cf] summary
  variant       A_before   A_after   B_after   forget       avg
  ------------------------------------------------------------
  plain          82.40%    44.17%    87.55%   +38.23%    65.86%
  tiers_only     82.10%    55.86%    86.30%   +26.24%    71.08%
  ewc_only       82.85%    63.40%    85.10%   +19.45%    74.25%
  tiers_ewc      82.70%    71.95%    85.62%   +10.75%    78.79%
```

(The numbers above are illustrative — actual values depend on epochs,
seeds, and `--ewc-lambda`.)

How to interpret:

- **`forget` decreasing across variants** = each component reduces
  catastrophic forgetting.
- **`B_after` not collapsing** = retention isn’t bought by failing on
  the new task.
- **`avg` is the headline metric** — it should be highest for the full
  method.

## Limitations / caveats

- The harness uses a small backbone (`d_model=192`) and short schedules;
  treat absolute numbers as **relative ablation signals**, not
  state-of-the-art.
- EWC strength is sensitive to `--ewc-lambda`; sweep
  `[100, 400, 800, 1600]` if the cross-domain numbers swing wildly.
- Fisher is estimated with the *post-Phase-A model*. If you use very
  short Phase A (1 epoch), the Fisher estimate will be noisy; bump
  `--n-fisher` or `--epochs-a` to compensate.
