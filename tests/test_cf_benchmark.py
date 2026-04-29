"""Synthetic integration test for `cf_benchmark.py`.

We deliberately avoid downloading CIFAR-10/SVHN here — that would make the
test slow and network-dependent. Instead we feed the benchmark's training
loop with random tensors via in-memory `Dataset`s and verify that all four
variants execute end-to-end and emit the expected metric keys.
"""
from __future__ import annotations

from pathlib import Path

import sys
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `cf_benchmark.py` lives at the package root; import after sys.path tweaks.
import cf_benchmark as cf  # type: ignore[import-not-found]
from nlv.model import VisionModelConfig


class _SynthImages(Dataset):
    def __init__(self, n: int, num_classes: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx])


def _loaders(num_classes: int, n: int = 32, batch: int = 8):
    train = DataLoader(_SynthImages(n, num_classes, seed=1), batch_size=batch, shuffle=True)
    val = DataLoader(_SynthImages(n, num_classes, seed=2), batch_size=batch, shuffle=False)
    return train, val


def _run(variant: str) -> dict:
    cfg = VisionModelConfig(
        num_classes=5, d_model=32, num_heads=4,
        num_slow=1, num_mid=1, num_fast=1,
        patch_size=4, img_size=32,
    )
    a_tr, a_va = _loaders(num_classes=5)
    b_tr, b_va = _loaders(num_classes=5)
    return cf.run_variant(
        variant,
        cfg=cfg,
        a_train=a_tr, a_val=a_va,
        b_train=b_tr, b_val=b_va,
        head_a="task_a", head_b="task_b",
        n_classes_a=5, n_classes_b=5,
        device=torch.device("cpu"),
        epochs_a=1, epochs_b=1,
        lr=1e-3, head_lr=1e-3,
        ewc_lambda=1.0, n_fisher=2,
        seed=0,
    )


def _assert_keys(report: dict) -> None:
    for k in ("variant", "acc_a_before", "acc_a_after", "acc_b_after",
              "forgetting", "average", "wallclock_sec"):
        assert k in report, f"missing key: {k}"


def test_plain_variant() -> None:
    _assert_keys(_run("plain"))


def test_tiers_only_variant() -> None:
    _assert_keys(_run("tiers_only"))


def test_ewc_only_variant() -> None:
    _assert_keys(_run("ewc_only"))


def test_tiers_ewc_variant() -> None:
    _assert_keys(_run("tiers_ewc"))
