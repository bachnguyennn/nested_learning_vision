"""
utils.py — training utilities ported from vit_m3_project/utils.py.
"""
from __future__ import annotations
import torch
import torchvision.transforms as T


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def topk_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return results


def get_cifar100_transforms(*, train: bool) -> T.Compose:
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_imagenet_transforms(*, train: bool, img_size: int = 224) -> T.Compose:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_ndim: dict[int, int] = {}
    for p in model.parameters():
        by_ndim[p.ndim] = by_ndim.get(p.ndim, 0) + p.numel()
    return {"total": total, "trainable": trainable, "by_ndim": by_ndim}


def print_model_summary(model: torch.nn.Module, config) -> None:
    info = count_parameters(model)
    print(f"\n{'═'*60}")
    print(f"  NestedVisionModel  |  {config.num_classes} classes")
    print(f"  d_model={config.d_model}  heads={config.num_heads}  "
          f"patch={config.patch_size}  img={config.img_size}")
    print(f"  Tiers: Slow×{config.num_slow} Mid×{config.num_mid} Fast×{config.num_fast}")
    print(f"  Total params   : {info['total']:,}")
    print(f"  Trainable      : {info['trainable']:,}")
    by_ndim = info['by_ndim']
    n2d = sum(v for k, v in by_ndim.items() if k >= 2)
    n1d = by_ndim.get(1, 0)
    pct = 100.0 * n2d / max(info['total'], 1)
    print(f"  ≥2D params     : {n2d:,}  ({pct:.1f}% — M3 eligible)")
    if n1d > 0:
        print(f"  [WARN] 1D params: {n1d:,} — check for bias or LayerNorm leaks!")
    else:
        print(f"  1D params      : 0 ✓  (100% M3 compatible)")
    print(f"{'═'*60}\n")
