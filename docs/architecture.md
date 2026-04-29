# Architecture refresher

A short pointer to the key building blocks. For deeper detail, read
the docstrings in each file.

## Model — `src/nlv/model.py`

```text
Image [B, 3, H, W]
  └─ FlatPatchEmbed | ConvStemEmbed   → [B, N, D]
  └─ append zero CLS slot at position N → [B, N+1, D]
  └─ + pos_embed [1, N+1, D]
  └─ SLOW tier  (N × HOPEBlock)   ── update every 256 steps  (global scene)
  └─ MID  tier  (N × HOPEBlock)   ── update every 16 steps   (object parts)
  └─ FAST tier  (N × HOPEBlock)   ── update every step       (textures/edges)
  └─ RMSNorm (parameterless)
  └─ Read x[:, -1, :] (CLS slot)
  └─ Linear head → logits
```

Notable details:

- **No separate `cls_token` parameter.** The CLS slot is a zero-content
  token appended at position `N`; the CLS positional encoding lives in
  `pos_embed[-1]`. This keeps every parameter ≥ 2-D, satisfying M3.
- **`reset_memory_per_forward`** in `VisionModelConfig` decides whether
  the recurrent fast-weight memory inside each block is wiped each
  forward (default `True` for i.i.d. images). Set to `False` for
  streaming / TBPTT.

## HOPEBlock — `src/nlv/hope_block.py`

```text
x (B, S, D)
  → RMSNorm
  → qkv linear (bias=False)
  → split q, k, v ; QK L2-norm
  → ChunkedGatedDeltaNode
       Write: u_t = η(v - vhat) - β·vhat
       M_t   = M_{t-1} + Σ u_t ⊗ k_t
       Read:  o_t = M_t q_t
  → residual + o_proj
  → RMSNorm + GELU FFN (bias=False)
```

The chunked forward substitution is correct but `O(C)` in Python; this
is fine at CIFAR-scale (`chunk_size=16`, sequence 65) and is on the
**deferred-optimisation list** for larger inputs.

## Optimizers — `src/nlv/optim/`

- **`M3`**: Multi-scale Momentum Muon. Maintains `m1` (fast EMA), `m2`
  (slow EMA, refreshed every `slow_chunk` steps), and `v` (RMSprop
  denominator). Update: `p -= lr · (NS(m1) + α · NS(m2)) / (sqrt(v) + eps)`.
  Newton-Schulz runs in fp32 even when the model is in bf16/fp16.
- **`TieredOptimizerManager`**: builds three `M3` instances (Slow / Mid
  / Fast), each over a disjoint parameter group, driven by `LevelClock`.
  Cross-tier overlap is now refused at construction time.

## EWC

Used for continual-learning experiments only. Classical Kirkpatrick
form: estimate diagonal Fisher over `n_batches` of the previous task,
anchor weights, then add `λ · Σ F_i (θ_i − θ_i*)²` to the loss during
the next task. In `nested_learning_vision`, EWC is applied **only to the
slow tier** because that tier is updated rarely and is intended to hold
stable global features.

## Why “100% M3”

M3 = Newton-Schulz orthogonalisation. NS is mathematically defined for
matrices (≥ 2-D); 1-D parameters (biases, LayerNorm gamma/beta) cannot
be orthogonalised meaningfully. The project deliberately removes every
1-D parameter:

- `bias=False` on every Conv2d / Linear,
- parameterless `RMSNorm` (no gamma/beta),
- no `cls_token` (`[1, 1, D]` would be 3-D-but-degenerate),
- `pos_embed [1, N+1, D]` is reshaped to `[N+1, D]` by `_orthogonalize`.

`smoke_test.py` and `tests/test_manager.py` check this invariant.
