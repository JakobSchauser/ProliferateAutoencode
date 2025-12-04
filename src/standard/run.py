import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.simulation import SimConfig, train_division


def main():
    cfg = SimConfig()

    world = train_division(
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        cfg=cfg,
        lr=cfg.lr,
    )

    final_count = 0
    try:
        final_count = int(world.x.shape[0])
    except Exception:
        final_count = 0
    print("Training finished. Final world size after last reset:", final_count)


if __name__ == "__main__":
    main()
