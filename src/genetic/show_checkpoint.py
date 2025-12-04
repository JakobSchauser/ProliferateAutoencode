import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt

from genetic.simulation_genetic import GAConfig, _load_checkpoint, ParticleWorld
from NCAArchitecture import ParticleNCA


def load_best_from_checkpoint(resume_path: str) -> ParticleNCA:
    cfg = GAConfig(resume_path=resume_path)
    start_gen, population, history = _load_checkpoint(cfg)
    # Pick first individual from loaded population
    # Alternatively, re-evaluate and pick best; for speed, just use index 0
    model = population[0]
    print(f"Loaded checkpoint gen={start_gen}, population size={len(population)}, history len={len(history)}")
    return model


def rollout_and_show(model: ParticleNCA, steps: int = 300, dt: float = 0.1, device: str = "cpu", interval: int = 10):
    cfg = GAConfig(dt=dt, device=device)
    world = ParticleWorld(cfg, model)
    world.reset(n0=1)

    xs = []
    ts = []
    for t in range(steps):
        world.step()
        if (t + 1) % interval == 0:
            xs.append(world.x.detach().cpu())
            ts.append(t + 1)
            print(f"Step {t+1}/{steps} | cells={world.x.shape[0]}")

    # Plot snapshots every `interval`
    n = len(xs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), squeeze=False)
    for i, (snap, tstep) in enumerate(zip(xs, ts)):
        r = i // cols
        c = i % cols
        pos = snap.numpy()
        ax = axes[r][c]
        ax.scatter(pos[:, 0], pos[:, 1], s=4, alpha=0.7)
        ax.set_title(f"t={tstep} (n={pos.shape[0]})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
    # Hide any unused axes
    for j in range(n, rows*cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # Provide path to a checkpoint like checkpoints/ga_gen_50.pt
    resume_path = os.environ.get("GA_RESUME", os.path.join(os.path.dirname(__file__), "checkpoints", "ga_gen_200.pt"))
    model = load_best_from_checkpoint(resume_path)
    rollout_and_show(model, steps=200, dt=0.1, device="cpu", interval=20)


if __name__ == "__main__":
    main()
