#!/usr/bin/env python3
"""Quick test to verify imports work."""

try:
    from training.bc_trainer import pretrain
    from training.self_play import SelfPlayTrainer
    from training.bc_dataset import generate_bc_dataset
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
