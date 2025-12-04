import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from src.NCAArchitecture import ParticleNCA


@dataclass
class SimConfig:
    n_molecules: int = 16
    k: int = 16
    cutoff: float = 0.25
    dt: float = 0.1
    max_cells: int = 10000  # Allow overshoot beyond target
    device: str = "cpu"
    angle_sin_cos: bool = True
    positional_encoding: bool = True
    aggregate: str = "sum"
    # Training params
    epochs: int = 2000
    steps_per_epoch: int = 1000
    lr: float = 1e-3
    # Loss selection: 'proxy' (differentiable) or 'count' (non-differentiable)
    loss_type: str = "count"


class ParticleWorld:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ParticleNCA(
            molecule_dim=cfg.n_molecules,
            k=cfg.k,
            cutoff=cfg.cutoff,
            angle_sin_cos=cfg.angle_sin_cos,
            positional_encoding=cfg.positional_encoding,
            aggregate=cfg.aggregate,
        ).to(self.device)
        # State tensors
        self.x = None
        self.angle = None
        self.mol = None
        self.gen = None

    def reset(self, n0: int = 1):
        self.x = torch.zeros(n0, 2, device=self.device)
        self.angle = (torch.zeros(n0, 1, device=self.device))
        self.mol = torch.zeros(n0, self.cfg.n_molecules, device=self.device)
        self.gen = torch.zeros(n0, 1, device=self.device)

    @torch.no_grad()
    def _perform_divisions(self, divide_mask: torch.Tensor):
        # divide_mask: (N,) boolean
        if divide_mask.sum() == 0:
            return
        N = self.x.shape[0]
        to_divide_idx = torch.nonzero(divide_mask, as_tuple=False).squeeze(-1)
        num_new = int(to_divide_idx.numel())
        if num_new <= 0:
            return
        # Allow overshoot beyond max_cells
        sel = to_divide_idx
        # Create children duplicated from selected parents, with small jitter
        jitter_pos = 0.01
        jitter_ang = 0.01
        x_child = self.x[sel] + (torch.randn_like(self.x[sel]) * jitter_pos)
        angle_child = self.angle[sel] + (torch.randn_like(self.angle[sel]) * jitter_ang)
        mol_child = self.mol[sel]
        gen_child = self.gen[sel] + 1.0
        # Concatenate to world
        self.x = torch.cat([self.x, x_child], dim=0)
        self.angle = torch.cat([self.angle, angle_child], dim=0)
        self.mol = torch.cat([self.mol, mol_child], dim=0)
        self.gen = torch.cat([self.gen, gen_child], dim=0)

    def step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dxdy, dtheta, dmol, div_logit = self.model(self.x, self.angle, self.mol, self.gen)
        # Euler update
        self.x = self.x + dxdy * self.cfg.dt
        self.angle = self.angle + dtheta * self.cfg.dt
        self.mol = self.mol + dmol * self.cfg.dt
        # Division decision (soft threshold); training may use the logits directly
        divide = (torch.sigmoid(div_logit) > 0.5).squeeze(-1)
        # Perform divisions without grad
        self._perform_divisions(divide)
        return dxdy, dtheta, dmol, div_logit


# Loss function API (modular)
LossFn = Callable[[ParticleWorld, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]


def growth_proxy_loss(
    world: ParticleWorld,
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_count: int = 4096,
) -> torch.Tensor:
    # Differentiable proxy: drive divide_logit high when under target, low when over target.
    _, _, _, div_logit = outputs  # (N,1)
    prob = torch.sigmoid(div_logit).mean()  # scalar
    curr = float(world.x.shape[0]) if world.x is not None else 0.0
    target = float(target_count)
    desired = 1.0 if curr < target else 0.0
    desired_t = torch.tensor(desired, device=div_logit.device)
    return (prob - desired_t) ** 2

def count_target_loss(
    world: ParticleWorld,
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_count: int = 4096,
) -> torch.Tensor:
    # Pure distance to target count (non-differentiable w.r.t. model parameters)
    curr = float(world.x.shape[0]) if world.x is not None else 0.0
    target = float(target_count)
    loss_val = (curr - target) ** 2
    return torch.tensor(loss_val, dtype=torch.float32, device=world.device)


def make_optimizer(world: ParticleWorld, lr: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adam(world.model.parameters(), lr=lr)


def rollout_and_loss(world: ParticleWorld, steps: int, loss_fn: LossFn) -> torch.Tensor:
    total_loss = torch.zeros((), device=world.device)
    for _ in range(steps):
        outputs = world.step()
        total_loss = total_loss + loss_fn(world, outputs)
        # Allow overshoot beyond target
    return total_loss


def train_division(
    epochs: int = 2000,
    steps_per_epoch: int = 1000,
    cfg: Optional[SimConfig] = None,
    loss_fn: Optional[LossFn] = None,
    lr: float = 1e-3,
):
    cfg = cfg or SimConfig()
    world = ParticleWorld(cfg)
    world.reset(n0=1)
    optimizer = make_optimizer(world, lr=lr)
    if loss_fn is None:
        # Choose loss per config
        if cfg.loss_type == "proxy":
            def _lf(w: ParticleWorld, outputs):
                return growth_proxy_loss(w, outputs, target_count=cfg.max_cells)
            loss_fn = _lf
        elif cfg.loss_type == "count":
            def _lf(w: ParticleWorld, outputs):
                return count_target_loss(w, outputs, target_count=cfg.max_cells)
            loss_fn = _lf
        else:
            raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    # Determine print cadence: 100 updates across training
    print_every = max(1, epochs // 100)
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = rollout_and_loss(world, steps_per_epoch, loss_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world.model.parameters(), max_norm=1.0)
        optimizer.step()
        # Reset world at epoch boundaries to keep curriculum simple
        # Compute a simple accuracy proxy: how close we are to target count (clamped to 1.0)
        curr_count = world.x.shape[0] if world.x is not None else 0
        target = cfg.max_cells
        accuracy = curr_count / float(target)
        if (ep + 1) % print_every == 0 or (ep + 1) == epochs:
            print(f"Epoch {ep+1}/{epochs} | loss={loss.item():.4f} | acc={accuracy:.3f}")
        world.reset(n0=1)

    return world
