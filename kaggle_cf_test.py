# ============================================================
# CELL 1 — GPU CHECK + INSTALL
# ============================================================
import subprocess, torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch: {torch.__version__}")
subprocess.run(["pip", "install", "torchvision", "-q"])


# ============================================================
# CELL 2 — PASTE ALL MODEL CODE (self-contained, no local files)
# ============================================================
import torch, torch.nn as nn, torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class ChunkedDeltaNode(nn.Module):
    def __init__(self, H, Dh, C=16):
        super().__init__()
        self.H, self.Dh, self.C = H, Dh, C
        self.M0 = nn.Parameter(torch.zeros(1, H, Dh, Dh))

    def forward(self, q, k, v, beta, eta):
        B, S, H, Dh = k.shape
        k = F.normalize(k, dim=-1, eps=1e-6)
        q = F.normalize(q, dim=-1, eps=1e-6)
        M = self.M0.expand(B, -1, -1, -1).clone()
        C = self.C
        nc = (S + C - 1) // C
        pad = nc * C - S
        def _pad(t, z=Dh): return torch.cat([t, torch.zeros(B, pad, H, z, device=t.device, dtype=t.dtype)], 1) if pad else t
        def _padg(t):       return torch.cat([t, torch.zeros(B, pad, H, 1,  device=t.device, dtype=t.dtype)], 1) if pad else t
        k,v,q,beta,eta = _pad(k),_pad(v),_pad(q),_padg(beta),_padg(eta)
        Sp = nc*C
        msk = (torch.arange(Sp, device=k.device) < S).view(1,Sp,1,1).to(k.dtype)
        k,v,q,beta,eta = k*msk,v*msk,q*msk,beta*msk,eta*msk
        out = []
        eye = torch.eye(C, device=k.device, dtype=k.dtype).view(1,1,C,C)
        for n in range(nc):
            sl = slice(n*C,(n+1)*C)
            kh = k[:,sl].permute(0,2,1,3); vh=v[:,sl].permute(0,2,1,3)
            qh = q[:,sl].permute(0,2,1,3); bh=beta[:,sl].permute(0,2,1,3); eh=eta[:,sl].permute(0,2,1,3)
            r0 = torch.einsum("bhij,bhcj->bhci", M, kh)
            G  = torch.tril(torch.einsum("bhid,bhjd->bhij",kh,kh), diagonal=-1)
            lam= (eh+bh).squeeze(-1)
            L  = eye + G*lam.unsqueeze(-2)
            rhs= r0 + torch.einsum("bhij,bhjd->bhid",G,eh*vh)
            rows=[]
            for r in range(C):
                b = rhs[:,:,r:r+1,:]
                if r>0: b = b - torch.matmul(L[:,:,r:r+1,:r], torch.cat(rows,dim=2))
                rows.append(b/(L[:,:,r:r+1,r:r+1]+1e-8))
            vhat = torch.cat(rows,dim=2)
            u = eh*(vh-vhat)-bh*vhat
            M = M + torch.einsum("bhci,bhcj->bhij",u,kh)
            out.append(torch.einsum("bhij,bhcj->bhci",M,qh).permute(0,2,1,3))
        return torch.cat(out,1)[:,:S]


class HOPEBlock(nn.Module):
    def __init__(self, D, H=6, C=16, expansion=4):
        super().__init__()
        Dh = D // H
        self.H, self.Dh = H, Dh
        self.norm   = RMSNorm(D)
        self.qkv    = nn.Linear(D, D*3, bias=False)
        self.rates  = nn.Linear(D, H*2, bias=False)
        self.delta  = ChunkedDeltaNode(H, Dh, C)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.fn     = RMSNorm(D)
        self.ff     = nn.Sequential(nn.Linear(D, D*expansion, bias=False), nn.GELU(),
                                    nn.Linear(D*expansion, D, bias=False))

    def forward(self, x):
        B,S,D = x.shape; H,Dh = self.H, self.Dh
        res = x; x = self.norm(x)
        q,k,v = self.qkv(x).chunk(3,-1)
        q=q.view(B,S,H,Dh); k=k.view(B,S,H,Dh); v=v.view(B,S,H,Dh)
        r = torch.sigmoid(self.rates(x))
        beta=r[:,:,:H].unsqueeze(-1); eta=r[:,:,H:].unsqueeze(-1)
        out = self.delta(q,k,v,beta,eta).reshape(B,S,D)
        x = res + self.o_proj(out)
        return x + self.ff(self.fn(x))


class NestedVisionModel(nn.Module):
    """3-tier nested vision model. 100% M3 (zero 1D params)."""
    def __init__(self, num_classes=100, D=192, H=6, C=16,
                 n_slow=2, n_mid=2, n_fast=2, patch=4, img=32):
        super().__init__()
        N = (img//patch)**2
        self.patch_embed = nn.Conv2d(3, D, patch, stride=patch, bias=False)
        self.pos_embed   = nn.Parameter(torch.zeros(1, N+1, D))   # CLS at N, no cls_token param
        self.slow = nn.ModuleList([HOPEBlock(D,H,C) for _ in range(n_slow)])
        self.mid  = nn.ModuleList([HOPEBlock(D,H,C) for _ in range(n_mid)])
        self.fast = nn.ModuleList([HOPEBlock(D,H,C) for _ in range(n_fast)])
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, num_classes, bias=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        B,_,_,_ = x.shape
        D = self.pos_embed.shape[-1]
        x = self.patch_embed(x).flatten(2).transpose(1,2)          # [B,N,D]
        cls = torch.zeros(B,1,D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, cls], 1) + self.pos_embed
        for blk in self.slow: x = blk(x)
        for blk in self.mid:  x = blk(x)
        for blk in self.fast: x = blk(x)
        return self.head(self.norm(x[:,-1,:]))

print("Model code loaded ✓")
n1d = sum(p.numel() for p in NestedVisionModel().parameters() if p.ndim < 2)
assert n1d == 0, f"FAIL: {n1d} 1D params found!"
print(f"100% M3 check: 0 1D parameters ✓")


# ============================================================
# CELL 3 — M3 OPTIMIZER (self-contained)
# ============================================================
class M3(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 beta3=0.9, alpha=1.0, ns_steps=3, slow_chunk=100, eps=1e-8):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2,
                                      beta3=beta3, alpha=alpha,
                                      ns_steps=ns_steps, slow_chunk=slow_chunk, eps=eps))

    @staticmethod
    def _ns(G, steps, eps=1e-6):
        if G.ndim < 2: return G
        x = G.reshape(G.shape[0],-1).float()
        x = x / (torch.linalg.norm(x) + eps)
        for _ in range(steps):
            x = 0.5*x@(3*torch.eye(x.shape[1],device=x.device,dtype=x.dtype) - x.T@x)
        return x.reshape_as(G).to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr,b1,b2,b3,a,ns,sc,eps = (g[k] for k in
                ("lr","beta1","beta2","beta3","alpha","ns_steps","slow_chunk","eps"))
            for p in g["params"]:
                if p.grad is None: continue
                grad = p.grad
                s = self.state[p]
                if not s:
                    s["step"]=0; s["m1"]=torch.zeros_like(p)
                    s["m2"]=torch.zeros_like(p); s["v"]=torch.zeros_like(p)
                    s["buf"]=torch.zeros_like(p); s["o2"]=torch.zeros_like(p)
                s["step"] += 1
                s["m1"].mul_(b1).add_(grad,alpha=1-b1)
                s["v"].mul_(b2).addcmul_(grad,grad,value=1-b2)
                s["buf"].add_(grad)
                o1 = self._ns(s["m1"], ns, eps)
                update = (o1 + a*s["o2"]) / (s["v"].sqrt().add_(eps))
                p.add_(update, alpha=-lr)
                if sc > 0 and s["step"] % sc == 0:
                    s["m2"].mul_(b3).add_(s["buf"],alpha=1-b3)
                    s["buf"].zero_()
                    s["o2"] = self._ns(s["m2"], ns, eps)

print("M3 optimizer loaded ✓")


# ============================================================
# CELL 4 — DATA: SPLIT CIFAR-100 INTO TASK A (0–49) AND TASK B (50–99)
# ============================================================
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

train_tf = T.Compose([T.RandomCrop(32,padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(MEAN,STD)])
val_tf   = T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])

full_train = torchvision.datasets.CIFAR100("/kaggle/working", train=True,  download=True, transform=train_tf)
full_val   = torchvision.datasets.CIFAR100("/kaggle/working", train=False, download=True, transform=val_tf)

def make_split(ds, class_range):
    lo, hi = class_range
    idx = [i for i,(_,y) in enumerate(ds) if lo <= y < hi]
    return Subset(ds, idx)

# Task A: classes 0-49  |  Task B: classes 50-99
task_a_train = make_split(full_train, (0,  50))
task_b_train = make_split(full_train, (50,100))
task_a_val   = make_split(full_val,   (0,  50))
task_b_val   = make_split(full_val,   (50,100))

BS = 256
ld = lambda ds, shuf: DataLoader(ds, batch_size=BS, shuffle=shuf, num_workers=2, pin_memory=True)
loaders = dict(
    a_train=ld(task_a_train,True),  a_val=ld(task_a_val,False),
    b_train=ld(task_b_train,True),  b_val=ld(task_b_val,False),
)
print(f"Task A train: {len(task_a_train):,}  val: {len(task_a_val):,}")
print(f"Task B train: {len(task_b_train):,}  val: {len(task_b_val):,}")


# ============================================================
# CELL 5 — EWC + TRAINING UTILITIES
# ============================================================
import copy

def topk(logits, labels, k=1):
    _, pred = logits.topk(k, dim=1)
    return pred.t().eq(labels.view(1,-1).expand_as(pred.t())).float().sum().item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        correct += topk(out, y, 1)
        total   += y.size(0)
    model.train()
    return 100.0 * correct / total


class EWC:
    """Fisher-diagonal EWC on the slow-tier layers."""
    def __init__(self, model, lam=1000.0):
        self.lam = lam
        self._anchor = {}
        self._fisher = {}

    def snapshot(self, model, loader, device, n_batches=20):
        """Estimate Fisher from gradient variance on Task A data."""
        model.eval()
        self._anchor = {n: p.data.clone() for n,p in model.slow.named_parameters()}
        self._fisher = {n: torch.zeros_like(p) for n,p in model.slow.named_parameters()}
        crit = nn.CrossEntropyLoss()
        for i,(x,y) in enumerate(loader):
            if i >= n_batches: break
            x,y = x.to(device),y.to(device)
            model.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            for n,p in model.slow.named_parameters():
                if p.grad is not None:
                    self._fisher[n] += p.grad.data.pow(2)
        for k in self._fisher:
            self._fisher[k] /= n_batches
        model.train()
        print("EWC snapshot done.")

    def penalty(self, model):
        if not self._anchor: return torch.tensor(0.0)
        loss = 0.0
        dev = next(model.parameters()).device
        for n,p in model.slow.named_parameters():
            F_ = self._fisher[n].to(dev)
            a  = self._anchor[n].to(dev)
            loss += (F_ * (p - a).pow(2)).sum()
        return self.lam * loss


def train_one_epoch(model, loader, opt, device, ewc=None):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss = 0.0
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        if ewc is not None: loss = loss + ewc.penalty(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


print("Training utilities loaded ✓")


# ============================================================
# CELL 6 — BUILD BOTH MODELS
# ============================================================
DEVICE = torch.device("cuda")
D, H, PATCH = 192, 6, 4

# Model 1: Baseline — same architecture but treated as single tier (no EWC)
baseline = NestedVisionModel(num_classes=100, D=D, H=H, patch=PATCH).to(DEVICE)

# Model 2: Nested — 3-tier with EWC protecting the slow tier
nested = NestedVisionModel(num_classes=100, D=D, H=H, patch=PATCH).to(DEVICE)

# Initialise them identically so comparison is fair
nested.load_state_dict(copy.deepcopy(baseline.state_dict()))

total = sum(p.numel() for p in baseline.parameters())
print(f"Both models: {total:,} params  |  device: {DEVICE}")

# Both use M3 — the ONLY difference is EWC during Task B
def make_opt(model, lr=1e-3):
    return M3(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, beta1=0.9, beta2=0.999, beta3=0.9,
        alpha=1.0, ns_steps=3, slow_chunk=100,
    )

opt_base   = make_opt(baseline)
opt_nested = make_opt(nested)

ewc = EWC(nested, lam=500.0)   # only applied to the nested model

print("Models and optimizers ready ✓")


# ============================================================
# CELL 7 — PHASE 1: TRAIN BOTH ON TASK A (classes 0–49)
# ============================================================
EPOCHS_A = 30
results = {"baseline": {}, "nested": {}}

print(f"\n{'='*55}")
print(f"PHASE 1 — Training on Task A (classes 0–49)  [{EPOCHS_A} epochs]")
print(f"{'='*55}")

for epoch in range(EPOCHS_A):
    lb = train_one_epoch(baseline, loaders["a_train"], opt_base,   DEVICE)
    ln = train_one_epoch(nested,   loaders["a_train"], opt_nested,  DEVICE)
    if (epoch+1) % 5 == 0:
        acc_b = evaluate(baseline, loaders["a_val"], DEVICE)
        acc_n = evaluate(nested,   loaders["a_val"], DEVICE)
        print(f"  Ep {epoch+1:3d} | Baseline A={acc_b:.1f}%  Nested A={acc_n:.1f}%")

# Record Task A accuracy BEFORE Task B
pre_b  = evaluate(baseline, loaders["a_val"], DEVICE)
pre_n  = evaluate(nested,   loaders["a_val"], DEVICE)
results["baseline"]["task_a_before"] = pre_b
results["nested"]["task_a_before"]   = pre_n
print(f"\nTask A before Task B:  Baseline={pre_b:.2f}%  Nested={pre_n:.2f}%")


# ============================================================
# CELL 8 — EWC SNAPSHOT (run after Task A, before Task B)
# ============================================================
ewc.snapshot(nested, loaders["a_val"], DEVICE, n_batches=30)

# Save Task-A weights for manual forgetting metric
snap_base   = {n: p.data.clone() for n,p in baseline.named_parameters()}
snap_nested = {n: p.data.clone() for n,p in nested.named_parameters()}
print("Snapshot complete — ready for Task B fine-tuning.")


# ============================================================
# CELL 9 — PHASE 2: FINE-TUNE ON TASK B (classes 50–99)
# This is where catastrophic forgetting is induced.
# ============================================================
EPOCHS_B = 20

# Lower LR for fine-tuning
opt_base_b   = make_opt(baseline, lr=5e-4)
opt_nested_b = make_opt(nested,   lr=5e-4)

print(f"\n{'='*55}")
print(f"PHASE 2 — Fine-tuning on Task B (classes 50–99)  [{EPOCHS_B} epochs]")
print(f"{'='*55}")
print(f"{'Epoch':>6} | {'Base_A':>8} {'Base_B':>8} | {'Nest_A':>8} {'Nest_B':>8}")
print("-"*55)

for epoch in range(EPOCHS_B):
    train_one_epoch(baseline, loaders["b_train"], opt_base_b,   DEVICE, ewc=None)
    train_one_epoch(nested,   loaders["b_train"], opt_nested_b, DEVICE, ewc=ewc)

    if (epoch+1) % 2 == 0:
        ba = evaluate(baseline, loaders["a_val"], DEVICE)
        bb = evaluate(baseline, loaders["b_val"], DEVICE)
        na = evaluate(nested,   loaders["a_val"], DEVICE)
        nb = evaluate(nested,   loaders["b_val"], DEVICE)
        print(f"  {epoch+1:3d}   | {ba:>7.1f}% {bb:>7.1f}% | {na:>7.1f}% {nb:>7.1f}%")


# ============================================================
# CELL 10 — RESULTS: FORGETTING METRICS
# ============================================================
final_ba = evaluate(baseline, loaders["a_val"], DEVICE)
final_bb = evaluate(baseline, loaders["b_val"], DEVICE)
final_na = evaluate(nested,   loaders["a_val"], DEVICE)
final_nb = evaluate(nested,   loaders["b_val"], DEVICE)

results["baseline"]["task_a_after"]  = final_ba
results["baseline"]["task_b_final"]  = final_bb
results["nested"]["task_a_after"]    = final_na
results["nested"]["task_b_final"]    = final_nb

forgetting_base   = results["baseline"]["task_a_before"] - final_ba
forgetting_nested = results["nested"]["task_a_before"]   - final_na

print(f"\n{'='*55}")
print(f"  CATASTROPHIC FORGETTING RESULTS")
print(f"{'='*55}")
print(f"  {'Metric':<35} {'Baseline':>10} {'Nested':>10}")
print(f"  {'-'*55}")
print(f"  {'Task A acc (before Task B)':35} {results['baseline']['task_a_before']:>9.2f}% {results['nested']['task_a_before']:>9.2f}%")
print(f"  {'Task A acc (after Task B)':35} {final_ba:>9.2f}% {final_na:>9.2f}%")
print(f"  {'Task B acc (final)':35} {final_bb:>9.2f}% {final_nb:>9.2f}%")
print(f"  {'─'*55}")
print(f"  {'Forgetting ΔA (lower = better)':35} {forgetting_base:>+9.2f}% {forgetting_nested:>+9.2f}%")
print(f"  {'Forgetting reduction':35} {forgetting_base - forgetting_nested:>+9.2f}%")
print(f"{'='*55}")


# ============================================================
# CELL 11 — PLOT
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart: forgetting comparison
ax = axes[0]
labels = ["Baseline\n(no protection)", "Nested+EWC\n(slow tier protected)"]
forgetting = [forgetting_base, forgetting_nested]
colors = ["#e74c3c", "#2ecc71"]
bars = ax.bar(labels, forgetting, color=colors, width=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Catastrophic Forgetting on Task A\n(lower = better)", fontsize=13, fontweight="bold")
ax.set_ylabel("Task A Accuracy Drop (%)")
for bar, val in zip(bars, forgetting):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:+.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(min(0, min(forgetting)-5), max(forgetting)+8)

# Grouped bar: full accuracy breakdown
ax2 = axes[1]
x = np.arange(2)
w = 0.35
b1 = ax2.bar(x - w/2, [results["baseline"]["task_a_before"], results["baseline"]["task_a_after"]],
             w, label="Task A (before/after)", color=["#3498db","#e74c3c"])
b2 = ax2.bar(x + w/2, [results["nested"]["task_a_before"], results["nested"]["task_a_after"]],
             w, label="Nested", color=["#3498db","#2ecc71"])
ax2.set_xticks(x); ax2.set_xticklabels(["Baseline", "Nested+EWC"])
ax2.set_ylabel("Task A Accuracy (%)")
ax2.set_title("Task A Accuracy: Before vs After Task B", fontsize=13, fontweight="bold")
ax2.legend(["Before Task B", "After Task B (Baseline)", "Before Task B", "After Task B (Nested)"])

# Simpler legend
from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(color="#3498db", label="Task A before Task B"),
    Patch(color="#e74c3c", label="After Task B — Baseline"),
    Patch(color="#2ecc71", label="After Task B — Nested+EWC"),
])

plt.tight_layout()
plt.savefig("/kaggle/working/forgetting_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to /kaggle/working/forgetting_results.png")
