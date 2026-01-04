import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation_coarse_gradient import GradConfig, gradient_train


def main():
    cfg = GradConfig(
        lr=3e-4,
        train_steps=100,
        rollout_steps=32,
        dt=0.1,
        n_molecules=8,
        k=8,
        cutoff=0.1,
        cell_size=0.08,
        init_cells=64,
        device="cpu",
        grad_clip=1.0,
        min_separation=0.02,
        sep_weight=0.1,
        noise_eta=1e-4,
    )

    model, hist = gradient_train(cfg)
    print("Gradient training finished. Steps run:", len(hist))


if __name__ == "__main__":
    main()
