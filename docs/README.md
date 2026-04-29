# nested_learning_vision — Documentation Index

This folder documents the audit, the code fixes that followed, the new
test suite, and the catastrophic-forgetting benchmark design.

| Topic                                              | File                                                 |
|----------------------------------------------------|------------------------------------------------------|
| What was audited and which issues were found       | [audit_2026-04-29.md](audit_2026-04-29.md)           |
| What changed in the code (file-by-file)            | [changelog.md](changelog.md)                         |
| New unit-test suite (`tests/`)                     | [testing.md](testing.md)                             |
| New catastrophic-forgetting benchmark methodology  | [cf_benchmark.md](cf_benchmark.md)                   |
| Architecture refresher (tiers, EWC, M3)            | [architecture.md](architecture.md)                   |

## Quick links

- Source: `src/nlv/`
- Tests: `tests/` — run with `python -m pytest tests/ -q`
- Smoke test (no dataset): `python smoke_test.py`
- CIFAR-100 training: `python train_cifar100.py`
- Catastrophic-forgetting benchmark: `python cf_benchmark.py --task split_cifar10`

## Status

- 22 unit tests passing (`tests/`)
- `smoke_test.py` passing (5/5)
- All audit findings either fixed or explicitly documented as “by design”
