import math
from dataclasses import dataclass
from typing import Tuple

import torch

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from NCAArchitecture2 import ParticleNCA_edge
from biological_coarse_graining.coarse_grain import sample_positions_from_image


@dataclass
class GradConfig:
    lr: float = 1e-3
    train_steps: int = 200
    rollout_steps: int = 32
    dt: float = 0.1
    n_molecules: int = 8
    k: int = 8
    cutoff: float = 0.1
    cell_size: float = 0.1
    init_cells: int = 8
    device: str = "cpu"
    grad_clip: float = 1.0
    min_separation: float = 0.02
    sep_weight: float = 0.1
    noise_eta: float = 1e-4


class ParticleWorldGrad:
    def __init__(self, cfg: GradConfig, model: ParticleNCA_edge):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.x = None
        self.angle = None
        self.mol = None
        self.gen = None

    def reset(self, n0: int):
        self.x = torch.zeros(n0, 2, device=self.device)
        self.angle = torch.zeros(n0, 1, device=self.device)
        self.mol = torch.zeros(n0, self.cfg.n_molecules, device=self.device)
        self.gen = torch.zeros(n0, 1, device=self.device)

    def step(self):
        dxdy, dtheta, dmol, _ = self.model(self.x, self.angle, self.mol, self.gen)
        eta = float(self.cfg.noise_eta)
        self.x = self.x + (dxdy + torch.randn_like(dxdy) * eta) * self.cfg.dt
        self.angle = self.angle + (dtheta + torch.randn_like(dtheta) * eta) * self.cfg.dt
        self.mol = self.mol + (dmol + torch.randn_like(dmol) * eta) * self.cfg.dt
        self.gen = self.gen + 1.0


def gaussian_coverage(world: ParticleWorldGrad, cfg: GradConfig, level: int = 0) -> torch.Tensor:
    target_pts_np = sample_positions_from_image(level)
    if target_pts_np.shape[0] == 0:
        raise RuntimeError("No target points available for coverage calculation")

    target = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)
    pts = world.x
    if pts is None or pts.shape[0] == 0:
        return torch.tensor(0.0, device=cfg.device)

    d2 = torch.cdist(target, pts) ** 2
    sigma2 = max(1e-12, float(cfg.cell_size) ** 2)
    field = torch.exp(-d2 / (2.0 * sigma2))
    coverage = field.max(dim=1).values.mean()
    return coverage


def separation_penalty(world: ParticleWorldGrad, min_dist: float) -> torch.Tensor:
    pts = world.x
    if pts is None or pts.shape[0] <= 1:
        return torch.tensor(0.0, device=world.device)

    d = torch.cdist(pts, pts)
    eye = torch.eye(d.shape[0], device=pts.device) * 1e6
    nn = (d + eye).min(dim=1).values
    shortfall = torch.relu(min_dist - nn)
    return shortfall.mean()


def rollout_and_loss(cfg: GradConfig, model: ParticleNCA_edge) -> Tuple[torch.Tensor, torch.Tensor]:
    world = ParticleWorldGrad(cfg, model)
    world.reset(cfg.init_cells)

    for _ in range(cfg.rollout_steps):
        world.step()

    coverage = gaussian_coverage(world, cfg)
    # sep_pen = separation_penalty(world, cfg.min_separation)
    loss = -coverage #+ cfg.sep_weight * sep_pen
    return coverage, loss


def gradient_train(cfg: GradConfig):
    model = ParticleNCA_edge(
        molecule_dim=cfg.n_molecules,
        k=cfg.k,
        cutoff=cfg.cutoff,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    best_cov = -math.inf

    for step in range(cfg.train_steps):
        opt.zero_grad()
        coverage, loss = rollout_and_loss(cfg, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        cov_val = float(coverage.detach().cpu())
        history.append(cov_val)
        best_cov = max(best_cov, cov_val)

        if (step + 1) % 10 == 0:
            print(f"[grad] step {step + 1}/{cfg.train_steps} | coverage={cov_val:.4f} | loss={float(loss.detach().cpu()):.4f} | best={best_cov:.4f}")

    return model, history


if __name__ == "__main__":
    cfg = GradConfig()
    gradient_train(cfg)
