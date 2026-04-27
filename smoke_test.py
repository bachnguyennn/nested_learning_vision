"""
smoke_test.py
─────────────────────────────────────────────────────────────────────────────
Quick sanity check — verifies:
  1. Model builds without error
  2. Forward pass runs, output shape is correct
  3. ZERO 1D parameters (100% M3 guarantee)
  4. Backward pass runs, gradients flow to all tiers
  5. TieredOptimizerManager steps without error
  6. Checkpoint save/load round-trip

Run: python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nlv import (
    NestedVisionModel,
    VisionModelConfig,
    TieredOptimizerManager,
    TierConfig,
    auto_device,
)
from nlv.utils import count_parameters, print_model_summary


def main() -> None:
    device = auto_device()
    print(f"Device: {device}\n")

    # ── Build model ───────────────────────────────────────────────────────
    config = VisionModelConfig(
        num_classes=100, d_model=192, num_heads=6,
        num_slow=2, num_mid=2, num_fast=2,
        patch_size=4, img_size=32, chunk_size=16,
    )
    model = NestedVisionModel(config).to(device)
    print_model_summary(model, config)

    # ── Test 1: Zero 1D parameters ────────────────────────────────────────
    info = count_parameters(model)
    n1d  = info["by_ndim"].get(1, 0)
    assert n1d == 0, f"FAIL: {n1d} 1D parameters found — M3 contract broken!"
    print("✓ Test 1 PASSED: Zero 1D parameters (100% M3 compatible)\n")

    # ── Test 2: Forward pass ──────────────────────────────────────────────
    B = 4
    x = torch.randn(B, 3, 32, 32, device=device)
    logits = model(x)
    assert logits.shape == (B, 100), f"FAIL: expected [{B}, 100], got {logits.shape}"
    print(f"✓ Test 2 PASSED: Forward pass output shape {logits.shape}\n")

    # ── Test 3: Backward pass + gradient flow ─────────────────────────────
    labels = torch.randint(0, 100, (B,), device=device)
    loss   = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"FAIL: no gradient for: {no_grad[:5]}"
    print(f"✓ Test 3 PASSED: Gradients flow to all {info['trainable']:,} parameters\n")

    # ── Test 4: TieredOptimizerManager step ───────────────────────────────
    tier_configs = [
        TierConfig("slow", update_period=256, lr=1e-4, ns_steps=3, slow_chunk=100),
        TierConfig("mid",  update_period=16,  lr=5e-4, ns_steps=3, slow_chunk=100),
        TierConfig("fast", update_period=1,   lr=1e-3, ns_steps=3, slow_chunk=100),
    ]
    mgr = TieredOptimizerManager(model, tier_configs)

    # Simulate 20 steps — fast should step every time, mid at step 16, slow never
    model.zero_grad()
    for step in range(20):
        x = torch.randn(B, 3, 32, 32, device=device)
        logits = model(x)
        loss   = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        ran = mgr.step()
        mgr.zero_grad()
        mgr.tick()
        if step in (0, 15, 16, 19):
            print(f"   Step {step+1:2d}: ran tiers = {[k for k,v in ran.items() if v]}")
    print("✓ Test 4 PASSED: TieredOptimizerManager stepped without error\n")

    # ── Test 5: Checkpoint round-trip ────────────────────────────────────
    import tempfile, os
    sd_before = {k: v.clone() for k, v in model.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save({"model": model.state_dict(), "optimizer": mgr.state_dict()}, tmp)
    ckpt = torch.load(tmp, map_location=device)
    model.load_state_dict(ckpt["model"])
    os.unlink(tmp)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, sd_before[k]), f"FAIL: param {k} changed after reload"
    print("✓ Test 5 PASSED: Checkpoint save/load round-trip\n")

    print("━" * 50)
    print("ALL TESTS PASSED — nested_learning_vision is ready.")
    print("━" * 50)


if __name__ == "__main__":
    main()
