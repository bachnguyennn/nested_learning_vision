# nested_learning_vision

**3-Tier Nested Learning Vision Model — 100% M3 Optimizer**

Adapted from `nested_learning` (LLM) and `vit_m3_project` (single-tier ViT).  
This project brings the full **Slow / Mid / Fast** nested hierarchy to computer vision,
with no AdamW, no SGD, no hybrid — pure Newton-Schulz gradient orthogonalization.

---

## Architecture

```
Image [B, 3, H, W]
  │
  ├─ FlatPatchEmbed / ConvStemEmbed  (Conv2d, bias=False)
  │       → [B, N, D]
  │
  ├─ + pos_embed [1, N+1, D]  (CLS slot at position N, no separate cls_token param)
  │
  ├─ SLOW TIER  (2× HOPEBlock)  ─── update every 256 steps  (global scene)
  ├─ MID  TIER  (2× HOPEBlock)  ─── update every 16 steps   (object parts)
  ├─ FAST TIER  (2× HOPEBlock)  ─── update every step       (textures/edges)
  │
  ├─ RMSNorm (parameterless)
  ├─ Read x[:, -1, :]  (CLS position)
  └─ Linear head → [B, num_classes]
```

### 100% M3 Compatibility Guarantee

| Component | Shape | M3? |
|---|---|---|
| Conv2d (patch embed) | `[D, C, P, P]` → flatten | ✅ |
| Linear (qkv, ff, head) | `[D_out, D_in]` | ✅ |
| `initial_M` (DeltaNode) | `[1, H, Dh, Dh]` → flatten | ✅ |
| `pos_embed` | `[1, N+1, D]` → flatten | ✅ |
| RMSNorm | No parameters | ✅ |
| `cls_token` | **Does not exist** — absorbed into pos_embed | ✅ |
| Any `bias` | All layers use `bias=False` | ✅ |
| LayerNorm | **Does not exist** — replaced by parameterless RMSNorm | ✅ |
| **1D params** | **Zero** | ✅ |

---

## Project Structure

```
nested_learning_vision/
├── src/nlv/
│   ├── __init__.py          # public API
│   ├── model.py             # NestedVisionModel, VisionModelConfig
│   ├── hope_block.py        # ChunkedGatedDeltaNode, HOPEBlock, RMSNorm
│   ├── patch_embed.py       # FlatPatchEmbed, ConvStemEmbed
│   ├── levels.py            # LevelSpec, LevelClock (tier scheduling)
│   ├── device.py            # resolve_device, auto_device
│   ├── utils.py             # AverageMeter, topk_accuracy, transforms
│   └── optim/
│       ├── m3.py            # M3 optimizer (Newton-Schulz two-momentum)
│       └── manager.py       # TieredOptimizerManager (Slow/Mid/Fast M3)
├── train_cifar100.py        # Full CIFAR-100 training script
├── smoke_test.py            # 5-test sanity check
└── configs/
    └── cifar100_small.yaml  # Config reference
```

---

## Quickstart

```bash
# Smoke test (no dataset needed)
python smoke_test.py

# Train on CIFAR-100 (auto-downloads)
python train_cifar100.py

# Custom config
python train_cifar100.py \
    --epochs 200 \
    --batch 256 \
    --lr 1e-3 \
    --d-model 384 \
    --num-heads 8 \
    --num-slow 4 --num-mid 4 --num-fast 4 \
    --embed-type stem \
    --ewc-lambda 0.1

# Resume from checkpoint
python train_cifar100.py --resume checkpoints/best.pt
```

---

## Key Design Decisions

### CLS Token → Position Slot
Standard ViT uses `cls_token = nn.Parameter([1, 1, D])` — a 3D degenerate
parameter that breaks M3 (Newton-Schulz requires `shape[0] > 1` for meaningful
orthogonalization). Instead, we use a **zero-content slot** appended to the patch
sequence, and the CLS *position* is encoded by `pos_embed[-1]`. The DeltaNode
memory read generates the CLS *content* from the accumulated patch memories.

### Why EWC on the Slow Tier?
The Slow tier is updated only every 256 steps and encodes **global scene priors**
that should be stable across tasks. EWC penalizes large deviations from the
Fisher-weighted anchor, preventing forgetting while still allowing adaptation.
Set `--ewc-lambda 0.0` (default) to disable for single-task training.

### M3 Two-Momentum Algorithm
Each M3 instance maintains:
- `m1` — fast EMA of gradient (orthogonalized every step → `o1`)  
- `m2` — slow EMA accumulated every `slow_chunk` steps (orthogonalized → `o2`)
- `v` — second moment (RMSprop denominator)
- Update: `p -= lr * (o1 + alpha * o2) / (sqrt(v) + eps)`

---

## Tier Update Frequencies

| Tier | Period | Learns | Optimizer LR |
|---|---|---|---|
| Fast | Every step | Textures, edges, patch details | `lr` |
| Mid | Every 16 steps | Object parts, spatial relations | `lr × 0.5` |
| Slow | Every 256 steps | Global scene, categorical priors | `lr × 0.1` |
