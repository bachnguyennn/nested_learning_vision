# ╔══════════════════════════════════════════════════════════════════╗
# ║  Catastrophic Forgetting Test: CIFAR-10 → SVHN                  ║
# ║  Model: NestedVisionModel (3-tier HOPE, 100% M3)                ║
# ║  Paste each # CELL N block into a separate Kaggle notebook cell  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────────
# CELL 1 — Clone, install, set device
# ──────────────────────────────────────────────────────────────────
import subprocess, sys, torch, torch.nn as nn, copy

subprocess.run(["git", "clone",
    "https://github.com/YOUR_USERNAME/nested_learning_vision",
    "/kaggle/working/nlv"], check=True)

sys.path.insert(0, "/kaggle/working/nlv/src")

import nlv
print(f"nlv loaded from: {nlv.__file__}")
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")

DEVICE = torch.device("cuda")


# ──────────────────────────────────────────────────────────────────
# CELL 2 — ContinualVisionModel (shared backbone, swappable heads)
# ──────────────────────────────────────────────────────────────────
from nlv import NestedVisionModel, VisionModelConfig, TieredOptimizerManager, TierConfig

class ContinualVisionModel(nn.Module):
    """Shared HOPE backbone + per-task linear heads.

    Fixes applied vs earlier version:
      - backbone_out calls patch_embed() directly (FlatPatchEmbed already
        returns [B, N, D] — no second flatten/transpose needed)
      - blk(x) returns (output, state) tuple → must unpack with x, _ = blk(x)
    """
    def __init__(self, cfg: VisionModelConfig):
        super().__init__()
        base = NestedVisionModel(cfg)
        self.patch_embed = base.patch_embed   # FlatPatchEmbed → returns [B,N,D]
        self.pos_embed   = base.pos_embed     # nn.Parameter [1, N+1, D]
        self.slow_layers = base.slow_layers   # ModuleList[HOPEBlock]
        self.mid_layers  = base.mid_layers
        self.fast_layers = base.fast_layers
        self.norm        = base.norm          # parameterless RMSNorm
        self.heads       = nn.ModuleDict()    # task_name → Linear
        self._active     = None
        self.D           = cfg.d_model

    def add_head(self, name: str, num_classes: int):
        h = nn.Linear(self.D, num_classes, bias=False)
        nn.init.trunc_normal_(h.weight, std=0.02)
        self.heads[name] = h.to(next(self.parameters()).device)
        print(f"  Head '{name}' added ({num_classes} classes)")

    def use_head(self, name: str):
        self._active = name

    def backbone_out(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # FlatPatchEmbed already returns [B, N, D] — do NOT flatten/transpose again
        x = self.patch_embed(x)
        cls = torch.zeros(B, 1, self.D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, cls], dim=1) + self.pos_embed          # [B, N+1, D]
        # HOPEBlock.forward returns (output, state) — unpack with _
        for blk in self.slow_layers: x, _ = blk(x)
        for blk in self.mid_layers:  x, _ = blk(x)
        for blk in self.fast_layers: x, _ = blk(x)
        return self.norm(x[:, -1, :])                             # [B, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads[self._active](self.backbone_out(x))


# Build config — 32×32 images, works for both CIFAR-10 and SVHN
cfg = VisionModelConfig(
    num_classes=10,   # placeholder, real outputs come from heads
    d_model=192, num_heads=6,
    num_slow=2, num_mid=2, num_fast=2,
    patch_size=4, img_size=32,
)

baseline = ContinualVisionModel(cfg).to(DEVICE)
nested   = ContinualVisionModel(cfg).to(DEVICE)
# Make backbones identical so only EWC protection differs
nested.load_state_dict(copy.deepcopy(baseline.state_dict()))

for m in [baseline, nested]:
    m.add_head("cifar10", 10)   # Task A
    m.add_head("svhn",    10)   # Task B (digits 0–9, completely different domain)

total = sum(p.numel() for p in baseline.parameters())
print(f"\nBackbone params : {total:,}")
print("Both models initialised identically ✓")
print("Only difference: nested gets EWC penalty during Task B")


# ──────────────────────────────────────────────────────────────────
# CELL 3 — Data: CIFAR-10 (Task A) and SVHN (Task B)
# ──────────────────────────────────────────────────────────────────
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

BATCH = 256

cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
svhn_mean,    svhn_std    = (0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)

loaders = {
    "a_tr": DataLoader(
        torchvision.datasets.CIFAR10("/kaggle/working", train=True,  download=True,
            transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                 T.ToTensor(), T.Normalize(cifar10_mean, cifar10_std)])),
        batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True),
    "a_va": DataLoader(
        torchvision.datasets.CIFAR10("/kaggle/working", train=False, download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize(cifar10_mean, cifar10_std)])),
        batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True),
    "b_tr": DataLoader(
        torchvision.datasets.SVHN("/kaggle/working", split="train", download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize(svhn_mean, svhn_std)])),
        batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True),
    "b_va": DataLoader(
        torchvision.datasets.SVHN("/kaggle/working", split="test",  download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize(svhn_mean, svhn_std)])),
        batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True),
}

print(f"Task A — CIFAR-10 : {len(loaders['a_tr'].dataset):,} train / {len(loaders['a_va'].dataset):,} val")
print(f"Task B — SVHN     : {len(loaders['b_tr'].dataset):,} train / {len(loaders['b_va'].dataset):,} val")
print("\nDomain gap:")
print("  CIFAR-10 → natural objects, soft textures, organic shapes")
print("  SVHN     → street digits, sharp edges, high contrast, structured crops")


# ──────────────────────────────────────────────────────────────────
# CELL 4 — Optimizers
# ──────────────────────────────────────────────────────────────────

def make_mgr(model: ContinualVisionModel, lr: float = 1e-3) -> TieredOptimizerManager:
    """
    Wire backbone parameters to 3-tier M3 optimizers.
    Task heads are excluded here — they get their own AdamW (make_head_opt).
    TieredOptimizerManager._get_tier_params uses attribute names:
      slow_layers, mid_layers, fast_layers, patch_embed, pos_embed, norm
    ContinualVisionModel has all of these. 'head' attr lookup returns None → skipped safely.
    """
    return TieredOptimizerManager(model, [
        TierConfig("slow", update_period=256, lr=lr * 0.1, ns_steps=5,
                   slow_chunk=200, warmup_steps=50),
        TierConfig("mid",  update_period=16,  lr=lr * 0.5, ns_steps=3,
                   slow_chunk=100),
        TierConfig("fast", update_period=1,   lr=lr,       ns_steps=3,
                   slow_chunk=100),
    ])

def make_head_opt(model: ContinualVisionModel, task: str, lr: float = 1e-3):
    return torch.optim.AdamW(model.heads[task].parameters(), lr=lr, weight_decay=0.01)

print("Optimizer factory ready ✓")


# ──────────────────────────────────────────────────────────────────
# CELL 5 — Training utilities (evaluate, EWC, train_epoch)
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: ContinualVisionModel, loader: DataLoader, task: str) -> float:
    model.eval()
    model.use_head(task)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    model.train()
    return 100.0 * correct / total


class EWC:
    """Fisher-diagonal EWC — anchors the Slow tier after Task A."""

    def __init__(self, lam: float = 800.0):
        self.lam = lam
        self._anchor: dict = {}
        self._fisher: dict = {}

    def snapshot(self, model: ContinualVisionModel, loader: DataLoader,
                 task: str, n_batches: int = 40):
        """Estimate Fisher from Task A gradients, anchor slow-tier weights."""
        model.eval()
        model.use_head(task)
        self._anchor = {k: p.data.clone()       for k, p in model.slow_layers.named_parameters()}
        self._fisher = {k: torch.zeros_like(p)  for k, p in model.slow_layers.named_parameters()}
        crit = nn.CrossEntropyLoss()
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            model.zero_grad()
            crit(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            for k, p in model.slow_layers.named_parameters():
                if p.grad is not None:
                    self._fisher[k] += p.grad.pow(2) / n_batches
        model.train()
        print(f"EWC snapshot: {len(self._anchor)} slow-tier param tensors anchored.")

    def penalty(self, model: ContinualVisionModel) -> torch.Tensor:
        if not self._anchor:
            return torch.tensor(0.0, device=DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        for k, p in model.slow_layers.named_parameters():
            F_ = self._fisher[k].to(p.device)
            a  = self._anchor[k].to(p.device)
            loss = loss + (F_ * (p - a).pow(2)).sum()
        return self.lam * loss


def train_epoch(model: ContinualVisionModel, mgr: TieredOptimizerManager,
                head_opt, loader: DataLoader, task: str, ewc: EWC = None):
    model.train()
    model.use_head(task)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Zero grads at the START of every epoch
    mgr.zero_grad()
    head_opt.zero_grad()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        loss = crit(model(x), y)
        if ewc is not None:
            loss = loss + ewc.penalty(model)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step backbone tiers that are due this step
        for name, opt in mgr.optimizers.items():
            if mgr.clock.should_update(name):
                opt.step()
                mgr.clock.record_update(name)

        # Step task head every batch (uses AdamW, not M3)
        head_opt.step()

        # Zero for next batch
        mgr.zero_grad()
        head_opt.zero_grad()
        mgr.tick()


ewc = EWC(lam=800.0)
print("Utilities ready ✓")


# ──────────────────────────────────────────────────────────────────
# CELL 6 — Phase 1: Train both models on CIFAR-10 (Task A)
# ──────────────────────────────────────────────────────────────────
EPOCHS_A = 30

mgr_base_a  = make_mgr(baseline, lr=1e-3)
mgr_nest_a  = make_mgr(nested,   lr=1e-3)
hopt_base_a = make_head_opt(baseline, "cifar10")
hopt_nest_a = make_head_opt(nested,   "cifar10")

print(f"PHASE 1 — CIFAR-10  [{EPOCHS_A} epochs]\n")
print(f"{'Ep':>4} | {'Baseline':>9} | {'Nested':>9}")
print("-" * 30)

for ep in range(1, EPOCHS_A + 1):
    train_epoch(baseline, mgr_base_a, hopt_base_a, loaders["a_tr"], "cifar10")
    train_epoch(nested,   mgr_nest_a, hopt_nest_a, loaders["a_tr"], "cifar10")
    if ep % 5 == 0:
        ba = evaluate(baseline, loaders["a_va"], "cifar10")
        na = evaluate(nested,   loaders["a_va"], "cifar10")
        print(f"{ep:4d} | {ba:>8.2f}% | {na:>8.2f}%")

pre_base = evaluate(baseline, loaders["a_va"], "cifar10")
pre_nest = evaluate(nested,   loaders["a_va"], "cifar10")
print(f"\nCIFAR-10 before SVHN:  Baseline={pre_base:.2f}%  Nested={pre_nest:.2f}%")


# ──────────────────────────────────────────────────────────────────
# CELL 7 — EWC Snapshot (run AFTER Task A, BEFORE Task B)
# ──────────────────────────────────────────────────────────────────
ewc.snapshot(nested, loaders["a_va"], "cifar10", n_batches=40)
print("Slow-tier anchored. Ready for Task B.")


# ──────────────────────────────────────────────────────────────────
# CELL 8 — Phase 2: Fine-tune on SVHN (Task B)
#          Baseline: no protection  |  Nested: EWC on slow tier
# ──────────────────────────────────────────────────────────────────
EPOCHS_B = 20

mgr_base_b  = make_mgr(baseline, lr=5e-4)
mgr_nest_b  = make_mgr(nested,   lr=5e-4)
hopt_base_b = make_head_opt(baseline, "svhn", lr=1e-3)
hopt_nest_b = make_head_opt(nested,   "svhn", lr=1e-3)

print(f"PHASE 2 — SVHN  [{EPOCHS_B} epochs]\n")
print(f"{'Ep':>4} | {'Base A':>8} {'Base B':>8} | {'Nest A':>8} {'Nest B':>8}")
print("-" * 50)

for ep in range(1, EPOCHS_B + 1):
    train_epoch(baseline, mgr_base_b, hopt_base_b, loaders["b_tr"], "svhn", ewc=None)
    train_epoch(nested,   mgr_nest_b, hopt_nest_b, loaders["b_tr"], "svhn", ewc=ewc)
    if ep % 2 == 0:
        ba = evaluate(baseline, loaders["a_va"], "cifar10")
        bb = evaluate(baseline, loaders["b_va"], "svhn")
        na = evaluate(nested,   loaders["a_va"], "cifar10")
        nb = evaluate(nested,   loaders["b_va"], "svhn")
        print(f"{ep:4d} | {ba:>7.2f}% {bb:>7.2f}% | {na:>7.2f}% {nb:>7.2f}%")


# ──────────────────────────────────────────────────────────────────
# CELL 9 — Final results + plot
# ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

final_ba = evaluate(baseline, loaders["a_va"], "cifar10")
final_bb = evaluate(baseline, loaders["b_va"], "svhn")
final_na = evaluate(nested,   loaders["a_va"], "cifar10")
final_nb = evaluate(nested,   loaders["b_va"], "svhn")

forgot_base = pre_base - final_ba
forgot_nest = pre_nest - final_na

print(f"\n{'='*60}")
print(f"  CIFAR-10 → SVHN  |  Catastrophic Forgetting Results")
print(f"{'='*60}")
print(f"  {'':34} {'Baseline':>10} {'Nested+EWC':>10}")
print(f"  {'CIFAR-10 acc before SVHN':34} {pre_base:>9.2f}% {pre_nest:>9.2f}%")
print(f"  {'CIFAR-10 acc after  SVHN':34} {final_ba:>9.2f}% {final_na:>9.2f}%")
print(f"  {'SVHN acc (new task)':34} {final_bb:>9.2f}% {final_nb:>9.2f}%")
print(f"  {'Forgetting ΔA (lower=better)':34} {forgot_base:>+9.2f}% {forgot_nest:>+9.2f}%")
print(f"  {'Forgetting reduction':34} {forgot_base - forgot_nest:>+9.2f}%")
print(f"{'='*60}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: forgetting bar
colors = ["#e74c3c", "#2ecc71"]
bars = ax1.bar(["Baseline\n(no protection)", "Nested+EWC\n(slow tier protected)"],
               [forgot_base, forgot_nest], color=colors, width=0.5)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_ylabel("CIFAR-10 Accuracy Drop (%)")
ax1.set_title("Catastrophic Forgetting on CIFAR-10\nafter learning SVHN", fontweight="bold")
for b, v in zip(bars, [forgot_base, forgot_nest]):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:+.1f}%",
             ha="center", fontsize=12, fontweight="bold")

# Right: before vs after
x = np.arange(2)
w = 0.35
ax2.bar(x - w/2, [pre_base, pre_nest],   w, label="Before SVHN", color="#3498db")
ax2.bar(x + w/2, [final_ba, final_na],   w, label="After SVHN",
        color=["#e74c3c", "#2ecc71"])
ax2.set_xticks(x)
ax2.set_xticklabels(["Baseline", "Nested+EWC"])
ax2.set_ylabel("CIFAR-10 Accuracy (%)")
ax2.set_ylim(0, 100)
ax2.set_title("CIFAR-10 Retention After Learning SVHN", fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig("/kaggle/working/cf_cifar10_svhn.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → /kaggle/working/cf_cifar10_svhn.png")
