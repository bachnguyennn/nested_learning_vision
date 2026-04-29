"""
cf_benchmark.py
─────────────────────────────────────────────────────────────────────────────
A self-contained, ablation-aware catastrophic forgetting harness for
`nested_learning_vision`.

What this script does that the kaggle scripts do **not**:

  1. Ablation matrix — four configurations are run on the SAME backbone init
     to disentangle the contributions of the nested-tier scheduler and EWC:

       * plain        : single-optimizer, no tier scheduler, no EWC
       * tiers_only   : 3-tier M3 (slow/mid/fast), no EWC
       * ewc_only     : single-optimizer, EWC on slow_layers params
       * tiers_ewc    : 3-tier M3 + EWC on slow tier   (the full method)

  2. Two task pairs run independently:

       * `split_cifar10`   : CIFAR-10 classes 0–4 → 5–9     (same domain)
       * `cifar10_to_svhn` : CIFAR-10 → SVHN               (cross domain)

  3. Reports the three numbers that matter:

       * Forgetting Δ_A  = Acc_A(before B) − Acc_A(after B)   (lower = better)
       * Plasticity     = Acc_B(after B)                      (higher = better)
       * Average        = mean(Acc_A_after, Acc_B_after)      (overall metric)

  4. Classical Kirkpatrick EWC: Fisher diagonal averaged over `n_fisher`
     batches of Task-A data, anchored ONCE between tasks.

Usage
─────
    python cf_benchmark.py --task split_cifar10 --variant tiers_ewc \
        --epochs-a 5 --epochs-b 5 --device auto

    # Run all four variants for split_cifar10 (writes a JSON report):
    python cf_benchmark.py --task split_cifar10 --variant all \
        --epochs-a 5 --epochs-b 5 --output reports/cf_split.json

This script is intentionally **single-file** so it can be lifted into a
notebook or another machine without dragging the rest of the package along.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

# ── package imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlv import (
    NestedVisionModel,
    VisionModelConfig,
    TieredOptimizerManager,
    TierConfig,
    auto_device,
)
from nlv.optim.m3 import M3


# ── small helpers ────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    model.train()
    return 100.0 * correct / max(total, 1)


# ── continual model: shared backbone + per-task heads ───────────────────────

class ContinualVisionModel(nn.Module):
    """Shared HOPE backbone + per-task linear heads.

    The backbone is `NestedVisionModel` minus its single fixed head.  Heads
    are added per task (`add_head`) and selected via `use_head`.
    """

    def __init__(self, cfg: VisionModelConfig) -> None:
        super().__init__()
        base = NestedVisionModel(cfg)
        self.cfg = cfg
        self.patch_embed = base.patch_embed
        self.pos_embed = base.pos_embed
        self.slow_layers = base.slow_layers
        self.mid_layers = base.mid_layers
        self.fast_layers = base.fast_layers
        self.norm = base.norm
        self.heads = nn.ModuleDict()
        self._active: str | None = None
        self.D = cfg.d_model
        # remember whether to reset memory per forward (mirrors model.py)
        self.reset_memory_per_forward = cfg.reset_memory_per_forward

    def add_head(self, name: str, num_classes: int) -> None:
        h = nn.Linear(self.D, num_classes, bias=False)
        nn.init.trunc_normal_(h.weight, std=0.02)
        device = next(self.parameters()).device
        self.heads[name] = h.to(device)

    def use_head(self, name: str) -> None:
        if name not in self.heads:
            raise KeyError(f"unknown head '{name}'; have {list(self.heads.keys())}")
        self._active = name

    def _reset_memory(self) -> None:
        for layer in (*self.slow_layers, *self.mid_layers, *self.fast_layers):
            layer.reset_memory()

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = torch.zeros(B, 1, self.D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, cls], dim=1) + self.pos_embed
        if self.reset_memory_per_forward:
            self._reset_memory()
        for blk in self.slow_layers:
            x, _ = blk(x)
        for blk in self.mid_layers:
            x, _ = blk(x)
        for blk in self.fast_layers:
            x, _ = blk(x)
        return self.norm(x[:, -1, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._active is None:
            raise RuntimeError("call use_head(name) before forward()")
        return self.heads[self._active](self.backbone(x))


# ── classical Kirkpatrick EWC on the slow tier ──────────────────────────────

class EWC:
    """Fisher-diagonal EWC, anchored ONCE between tasks."""

    def __init__(self, lam: float) -> None:
        self.lam = lam
        self._anchor: dict[str, torch.Tensor] = {}
        self._fisher: dict[str, torch.Tensor] = {}

    def snapshot(
        self,
        model: ContinualVisionModel,
        loader: DataLoader,
        task_head: str,
        device: torch.device,
        n_batches: int = 40,
    ) -> None:
        was_training = model.training
        model.eval()
        model.use_head(task_head)
        self._anchor = {
            name: p.data.detach().clone()
            for name, p in model.slow_layers.named_parameters()
        }
        self._fisher = {
            name: torch.zeros_like(p.data)
            for name, p in model.slow_layers.named_parameters()
        }
        crit = nn.CrossEntropyLoss()
        seen = 0
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            for p in model.slow_layers.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            crit(model(x), y).backward()
            for name, p in model.slow_layers.named_parameters():
                if p.grad is not None:
                    self._fisher[name] += p.grad.detach().pow(2)
            seen += 1
        denom = max(seen, 1)
        self._fisher = {n: f / denom for n, f in self._fisher.items()}
        # tidy up
        for p in model.slow_layers.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if was_training:
            model.train()

    def penalty(self, model: ContinualVisionModel) -> torch.Tensor:
        if not self._anchor:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for name, p in model.slow_layers.named_parameters():
            if name not in self._anchor:
                continue
            f = self._fisher[name].to(p.device)
            a = self._anchor[name].to(p.device)
            loss = loss + (f * (p - a).pow(2)).sum()
        return self.lam * loss


# ── data builders ───────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.198, 0.201, 0.197)


def _cifar10(data_dir: str, train: bool) -> torchvision.datasets.CIFAR10:
    if train:
        tfm = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return torchvision.datasets.CIFAR10(
        data_dir, train=train, download=True, transform=tfm,
    )


def _svhn(data_dir: str, train: bool) -> torchvision.datasets.SVHN:
    if train:
        tfm = T.Compose([T.ToTensor(), T.Normalize(SVHN_MEAN, SVHN_STD)])
    else:
        tfm = T.Compose([T.ToTensor(), T.Normalize(SVHN_MEAN, SVHN_STD)])
    split = "train" if train else "test"
    return torchvision.datasets.SVHN(
        data_dir, split=split, download=True, transform=tfm,
    )


class _RemapTargets(torch.utils.data.Dataset):
    """Wrap a dataset and remap labels via a dict (old → new)."""

    def __init__(self, base: torch.utils.data.Dataset, mapping: dict[int, int]) -> None:
        self.base = base
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, self.mapping[int(y)]


def _split_cifar10(data_dir: str, classes: Iterable[int], train: bool) -> _RemapTargets:
    """Return a CIFAR-10 subset over `classes`, with labels remapped to 0..k-1."""
    full = _cifar10(data_dir, train=train)
    classes = list(classes)
    mapping = {c: i for i, c in enumerate(classes)}
    indices = [i for i, t in enumerate(full.targets) if int(t) in mapping]
    return _RemapTargets(Subset(full, indices), mapping)


def build_loaders(task: str, data_dir: str, batch: int, num_workers: int):
    """Return ((train_a, val_a, ncls_a), (train_b, val_b, ncls_b), labels)."""
    if task == "split_cifar10":
        a_tr = _split_cifar10(data_dir, range(0, 5), train=True)
        a_va = _split_cifar10(data_dir, range(0, 5), train=False)
        b_tr = _split_cifar10(data_dir, range(5, 10), train=True)
        b_va = _split_cifar10(data_dir, range(5, 10), train=False)
        ncls_a = ncls_b = 5
        labels = ("cifar10_0_4", "cifar10_5_9")
    elif task == "cifar10_to_svhn":
        a_tr = _cifar10(data_dir, train=True)
        a_va = _cifar10(data_dir, train=False)
        b_tr = _svhn(data_dir, train=True)
        b_va = _svhn(data_dir, train=False)
        ncls_a = ncls_b = 10
        labels = ("cifar10", "svhn")
    else:
        raise ValueError(f"unknown task '{task}'")

    pin = torch.cuda.is_available()
    common = dict(batch_size=batch, num_workers=num_workers, pin_memory=pin)
    return (
        (DataLoader(a_tr, shuffle=True, **common),
         DataLoader(a_va, shuffle=False, **common),
         ncls_a),
        (DataLoader(b_tr, shuffle=True, **common),
         DataLoader(b_va, shuffle=False, **common),
         ncls_b),
        labels,
    )


# ── ablation factory ────────────────────────────────────────────────────────

VARIANTS = ("plain", "tiers_only", "ewc_only", "tiers_ewc")


def _backbone_params(model: ContinualVisionModel) -> list[nn.Parameter]:
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for module in (
        model.patch_embed,
        model.slow_layers,
        model.mid_layers,
        model.fast_layers,
    ):
        for p in module.parameters():
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
    if id(model.pos_embed) not in seen:
        out.append(model.pos_embed)
        seen.add(id(model.pos_embed))
    return out


@dataclass
class TrainingHandles:
    """Bundle the optimizer(s) and head optimizer for one variant."""

    variant: str
    backbone_optimizers: list  # either [M3] or list of M3 (one per tier)
    head_optimizers: dict[str, torch.optim.Optimizer]
    tiered: TieredOptimizerManager | None
    ewc: EWC | None

    def step_backbone(self, finite: bool) -> dict[str, bool]:
        ran: dict[str, bool] = {}
        if not finite:
            return ran
        if self.tiered is not None:
            for name, opt in self.tiered.optimizers.items():
                if self.tiered.clock.should_update(name):
                    opt.step()
                    self.tiered.clock.record_update(name)
                    ran[name] = True
        else:
            for opt in self.backbone_optimizers:
                opt.step()
            ran["all"] = True
        return ran

    def zero_backbone(self) -> None:
        if self.tiered is not None:
            self.tiered.zero_grad()
        else:
            for opt in self.backbone_optimizers:
                opt.zero_grad(set_to_none=True)

    def tick(self) -> None:
        if self.tiered is not None:
            self.tiered.tick()


def make_handles(
    variant: str,
    model: ContinualVisionModel,
    *,
    lr: float,
    head_lr: float,
    ewc_lambda: float,
) -> TrainingHandles:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}', expected one of {VARIANTS}")

    use_tiers = variant in ("tiers_only", "tiers_ewc")
    use_ewc = variant in ("ewc_only", "tiers_ewc")

    if use_tiers:
        tiered = TieredOptimizerManager(model, [
            TierConfig("slow", update_period=256, lr=lr * 0.1, ns_steps=5,
                       slow_chunk=200, warmup_steps=50),
            TierConfig("mid",  update_period=16,  lr=lr * 0.5, ns_steps=3,
                       slow_chunk=100),
            TierConfig("fast", update_period=1,   lr=lr,       ns_steps=3,
                       slow_chunk=100),
        ])
        backbone_optimizers: list = []
    else:
        tiered = None
        backbone_optimizers = [M3(_backbone_params(model), lr=lr, ns_steps=3)]

    head_optimizers = {
        name: torch.optim.AdamW(h.parameters(), lr=head_lr, weight_decay=0.01)
        for name, h in model.heads.items()
    }
    ewc = EWC(lam=ewc_lambda) if use_ewc else None
    return TrainingHandles(
        variant=variant,
        backbone_optimizers=backbone_optimizers,
        head_optimizers=head_optimizers,
        tiered=tiered,
        ewc=ewc,
    )


# ── training / evaluation loops ─────────────────────────────────────────────

def train_one_epoch(
    model: ContinualVisionModel,
    loader: DataLoader,
    handles: TrainingHandles,
    head_name: str,
    device: torch.device,
    *,
    use_amp: bool,
    apply_ewc: bool,
) -> float:
    model.train()
    model.use_head(head_name)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    head_opt = handles.head_optimizers[head_name]
    losses: list[float] = []
    amp_dtype = torch.bfloat16
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)
            if apply_ewc and handles.ewc is not None:
                loss = loss + handles.ewc.penalty(model)

        head_opt.zero_grad(set_to_none=True)
        handles.zero_backbone()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = bool(torch.isfinite(gn))
        handles.step_backbone(finite)
        if finite:
            head_opt.step()
        handles.tick()
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def run_variant(
    variant: str,
    *,
    cfg: VisionModelConfig,
    a_train: DataLoader,
    a_val: DataLoader,
    b_train: DataLoader,
    b_val: DataLoader,
    head_a: str,
    head_b: str,
    n_classes_a: int,
    n_classes_b: int,
    device: torch.device,
    epochs_a: int,
    epochs_b: int,
    lr: float,
    head_lr: float,
    ewc_lambda: float,
    n_fisher: int,
    seed: int,
) -> dict[str, float]:
    seed_everything(seed)
    model = ContinualVisionModel(cfg).to(device)
    model.add_head(head_a, n_classes_a)
    model.add_head(head_b, n_classes_b)
    handles = make_handles(
        variant, model, lr=lr, head_lr=head_lr, ewc_lambda=ewc_lambda,
    )
    use_amp = (device.type == "cuda")

    # Phase A
    t0 = time.time()
    for ep in range(epochs_a):
        train_one_epoch(
            model, a_train, handles, head_a, device,
            use_amp=use_amp, apply_ewc=False,
        )
    acc_a_before = evaluate_with_head(model, a_val, head_a, device)

    # Snapshot Fisher between phases for EWC variants
    if handles.ewc is not None:
        handles.ewc.snapshot(model, a_train, head_a, device, n_batches=n_fisher)

    # Phase B
    for ep in range(epochs_b):
        train_one_epoch(
            model, b_train, handles, head_b, device,
            use_amp=use_amp, apply_ewc=True,
        )
    acc_a_after = evaluate_with_head(model, a_val, head_a, device)
    acc_b_after = evaluate_with_head(model, b_val, head_b, device)
    elapsed = time.time() - t0

    return {
        "variant": variant,
        "acc_a_before": acc_a_before,
        "acc_a_after": acc_a_after,
        "acc_b_after": acc_b_after,
        "forgetting": acc_a_before - acc_a_after,
        "average": 0.5 * (acc_a_after + acc_b_after),
        "wallclock_sec": elapsed,
    }


def evaluate_with_head(
    model: ContinualVisionModel, loader: DataLoader, head: str, device: torch.device,
) -> float:
    model.use_head(head)
    return evaluate(model, loader, device)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Catastrophic-forgetting benchmark")
    p.add_argument("--task", choices=["split_cifar10", "cifar10_to_svhn"], required=True)
    p.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs-a", type=int, default=5)
    p.add_argument("--epochs-b", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--ewc-lambda", type=float, default=800.0)
    p.add_argument("--n-fisher", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None,
                   help="optional path to a JSON report")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--num-heads", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = auto_device() if args.device == "auto" else torch.device(args.device)
    print(f"[cf] task={args.task} device={device}")

    (a_loaders, b_loaders, head_labels) = build_loaders(
        args.task, args.data_dir, batch=args.batch, num_workers=args.num_workers,
    )
    a_train, a_val, n_a = a_loaders
    b_train, b_val, n_b = b_loaders
    head_a, head_b = head_labels

    cfg = VisionModelConfig(
        num_classes=max(n_a, n_b),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_slow=2, num_mid=2, num_fast=2,
        patch_size=4, img_size=32,
    )
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    results: list[dict] = []
    for v in variants:
        print(f"\n[cf] === running variant: {v} ===")
        r = run_variant(
            v,
            cfg=cfg,
            a_train=a_train, a_val=a_val,
            b_train=b_train, b_val=b_val,
            head_a=head_a, head_b=head_b,
            n_classes_a=n_a, n_classes_b=n_b,
            device=device,
            epochs_a=args.epochs_a, epochs_b=args.epochs_b,
            lr=args.lr, head_lr=args.head_lr,
            ewc_lambda=args.ewc_lambda, n_fisher=args.n_fisher,
            seed=args.seed,
        )
        results.append(r)
        print(
            f"  variant={v:<10} "
            f"A_before={r['acc_a_before']:.2f}% "
            f"A_after={r['acc_a_after']:.2f}% "
            f"B_after={r['acc_b_after']:.2f}% "
            f"forget={r['forgetting']:+.2f}% "
            f"avg={r['average']:.2f}% "
            f"({r['wallclock_sec']:.1f}s)"
        )

    print("\n[cf] summary")
    print(f"  {'variant':<12} {'A_before':>9} {'A_after':>9} {'B_after':>9} "
          f"{'forget':>8} {'avg':>8}")
    print("  " + "-" * 60)
    for r in results:
        print(
            f"  {r['variant']:<12} "
            f"{r['acc_a_before']:>8.2f}% "
            f"{r['acc_a_after']:>8.2f}% "
            f"{r['acc_b_after']:>8.2f}% "
            f"{r['forgetting']:>+7.2f}% "
            f"{r['average']:>7.2f}%"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {"task": args.task, "args": vars(args), "results": results},
                f, indent=2,
            )
        print(f"\n[cf] wrote {out}")


if __name__ == "__main__":
    main()
