"""
train_cifar100.py
─────────────────────────────────────────────────────────────────────────────
Training script for NestedVisionModel on CIFAR-100.
100% M3 — no AdamW, no SGD, no hybrid.

3-Tier update schedule:
    Fast  → every step        (patch_embed, pos_embed, fast_layers, head)
    Mid   → every 16 steps    (mid_layers)
    Slow  → every 256 steps   (slow_layers, with optional EWC)

Usage:
    python train_cifar100.py
    python train_cifar100.py --epochs 200 --lr 1e-3 --batch 256
    python train_cifar100.py --resume checkpoints/step_001000.pt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader

# ── local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nlv import (
    NestedVisionModel,
    VisionModelConfig,
    TieredOptimizerManager,
    TierConfig,
    auto_device,
)
from nlv.utils import (
    AverageMeter,
    topk_accuracy,
    get_cifar100_transforms,
    print_model_summary,
    count_parameters,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested Vision — CIFAR-100 (100% M3)")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch",        type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--data-dir",     type=str,   default="./data")
    p.add_argument("--ckpt-dir",     type=str,   default="./checkpoints")
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--d-model",      type=int,   default=192)
    p.add_argument("--num-heads",    type=int,   default=6)
    p.add_argument("--num-slow",     type=int,   default=2)
    p.add_argument("--num-mid",      type=int,   default=2)
    p.add_argument("--num-fast",     type=int,   default=2)
    p.add_argument("--chunk-size",   type=int,   default=16)
    p.add_argument("--patch-size",   type=int,   default=4)
    p.add_argument("--embed-type",   type=str,   default="flat", choices=["flat", "stem"])
    # Tier update cadences
    p.add_argument("--mid-period",   type=int,   default=16)
    p.add_argument("--slow-period",  type=int,   default=256)
    # EWC regularization on slow tier
    p.add_argument("--ewc-lambda",   type=float, default=0.0,
                   help="EWC penalty weight on slow tier (0 = disabled)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# EWC (Elastic Weight Consolidation) — protects Slow tier
# ─────────────────────────────────────────────────────────────────────────────

class EWC:
    """Lightweight EWC penalty on a set of parameters.

    After calling .snapshot(), the Fisher information is estimated from the
    current gradient and the penalty is ||θ - θ*||²_F (diagonal Fisher).
    """
    def __init__(self, model: NestedVisionModel, lam: float) -> None:
        self.model = model
        self.lam   = lam
        self._anchors: dict[str, torch.Tensor] = {}
        self._fishers: dict[str, torch.Tensor] = {}

    def snapshot(self) -> None:
        """Save current slow-tier weights and their Fisher diagonal."""
        self._anchors.clear()
        self._fishers.clear()
        for name, p in self.model.slow_layers.named_parameters():
            self._anchors[name] = p.data.clone()
            if p.grad is not None:
                self._fishers[name] = p.grad.data.pow(2).clone()
            else:
                self._fishers[name] = torch.ones_like(p.data)

    def penalty(self) -> torch.Tensor:
        """Return scalar EWC penalty (call before optimizer.step)."""
        if not self._anchors:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, p in self.model.slow_layers.named_parameters():
            if name in self._anchors:
                f   = self._fishers.get(name, torch.ones_like(p))
                diff = p - self._anchors[name].to(p.device)
                loss = loss + (f.to(p.device) * diff.pow(2)).sum()
        return self.lam * loss


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(args: argparse.Namespace, device: torch.device):
    train_ds = dsets.CIFAR100(args.data_dir, train=True,  download=True,
                               transform=get_cifar100_transforms(train=True))
    val_ds   = dsets.CIFAR100(args.data_dir, train=False, download=True,
                               transform=get_cifar100_transforms(train=False))
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: NestedVisionModel, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()
    use_amp = (device.type == "cuda")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        top1, top5 = topk_accuracy(logits.float(), labels, topk=(1, 5))
        n = images.size(0)
        loss_m.update(loss.item(), n)
        top1_m.update(top1, n)
        top5_m.update(top5, n)
    model.train()
    return loss_m.avg, top1_m.avg, top5_m.avg


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path, model: NestedVisionModel, mgr: TieredOptimizerManager,
    *, epoch: int, step: int, top1: float,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch":       epoch,
        "step":        step,
        "top1":        top1,
        "model":       model.state_dict(),
        "optimizer":   mgr.state_dict(),
    }, tmp)
    os.replace(tmp, path)
    print(f"  [ckpt] Saved {path}")


def load_checkpoint(
    path: str, model: NestedVisionModel, mgr: TieredOptimizerManager, device: torch.device,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    mgr.load_state_dict(ckpt["optimizer"])
    epoch = int(ckpt.get("epoch", 0))
    step  = int(ckpt.get("step",  0))
    top1  = float(ckpt.get("top1", 0.0))
    print(f"  [ckpt] Resumed from {path} (epoch={epoch}, step={step}, top1={top1:.2f}%)")
    return epoch, step


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = auto_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
    print(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    config = VisionModelConfig(
        num_classes = 100,
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        chunk_size  = args.chunk_size,
        num_slow    = args.num_slow,
        num_mid     = args.num_mid,
        num_fast    = args.num_fast,
        patch_size  = args.patch_size,
        img_size    = 32,
        embed_type  = args.embed_type,
    )
    model = NestedVisionModel(config).to(device)
    print_model_summary(model, config)

    # ── Verify 100% M3 (zero 1D params) ───────────────────────────────────
    info = count_parameters(model)
    n1d  = info["by_ndim"].get(1, 0)
    assert n1d == 0, (
        f"100% M3 violated: found {n1d} 1D parameters! "
        "Check for bias=True, LayerNorm with affine=True, or standalone nn.Parameter([D])."
    )

    # ── 3-Tier M3 Optimizers ──────────────────────────────────────────────
    # Decay learning rates by tier: Slow gets smallest lr (rarely updates)
    tier_configs = [
        TierConfig("slow", update_period=args.slow_period, lr=args.lr * 0.1,
                   ns_steps=5, slow_chunk=200, warmup_steps=100),
        TierConfig("mid",  update_period=args.mid_period,  lr=args.lr * 0.5,
                   ns_steps=3, slow_chunk=100),
        TierConfig("fast", update_period=1,                lr=args.lr,
                   ns_steps=3, slow_chunk=100),
    ]
    mgr = TieredOptimizerManager(model, tier_configs)

    # ── EWC (optional, slow tier protection) ─────────────────────────────
    ewc = EWC(model, lam=args.ewc_lambda) if args.ewc_lambda > 0 else None

    # ── Criterion + AMP ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp   = (device.type == "cuda")
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── LR scheduler (cosine, attached to fast optimizer) ─────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mgr.optimizers["fast"], T_max=args.epochs, eta_min=1e-5
    )

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(args, device)
    print(f"CIFAR-100: {len(train_loader.dataset):,} train / "
          f"{len(val_loader.dataset):,} val\n")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, mgr, device)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ────────────────────────────────────────────────────
    model.train()
    best_top1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        t0     = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)
                # Add EWC penalty if enabled
                if ewc is not None:
                    loss = loss + ewc.penalty()

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(mgr.optimizers["fast"])

            # Clip globally
            all_params = [p for p in model.parameters() if p.grad is not None]
            grad_norm  = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            if not torch.isnan(grad_norm):
                # ── Step all tiers due this step ──────────────────────────
                # (M3 reads .grad directly via optimizer.step())
                # We manually handle tier stepping since TieredOptimizerManager
                # uses its own grad reading — call scaler.step per optimizer.
                ran = {}
                for tier_name, opt in mgr.optimizers.items():
                    if mgr.clock.should_update(tier_name):
                        scaler.step(opt)
                        mgr.clock.record_update(tier_name)
                        ran[tier_name] = True
            else:
                print(f"  [WARN] ep{epoch+1} step{global_step}: NaN grad — skip")

            scaler.update()
            mgr.zero_grad()
            mgr.tick()
            global_step += 1

            # EWC snapshot on slow update
            if ewc is not None and ran.get("slow"):
                ewc.snapshot()

            # Metrics
            top1, _ = topk_accuracy(logits.float().detach(), labels)
            n = images.size(0)
            loss_m.update(loss.item(), n)
            top1_m.update(top1, n)

            if global_step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Ep {epoch+1:3d} | Step {global_step:6d} | "
                      f"Loss {loss_m.avg:.4f} | Train-Top1 {top1_m.avg:.1f}% | "
                      f"{elapsed:.1f}s")

        # ── Validation ────────────────────────────────────────────────────
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device)
        print(f"\n  ── Epoch {epoch+1}/{args.epochs} ──")
        print(f"     Val Loss  : {val_loss:.4f}")
        print(f"     Top-1     : {top1:.2f}%   Top-5: {top5:.2f}%")
        print(f"     Clock     : {mgr.clock.stats()}\n")

        scheduler.step()

        # Save checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}_top1_{top1:.1f}.pt"
        save_checkpoint(ckpt_path, model, mgr, epoch=epoch+1, step=global_step, top1=top1)
        if top1 > best_top1:
            best_top1 = top1
            save_checkpoint(ckpt_dir / "best.pt", model, mgr,
                            epoch=epoch+1, step=global_step, top1=top1)

    print(f"\nTraining complete. Best Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
