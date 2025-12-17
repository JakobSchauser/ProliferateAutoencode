import io
import os
from pyexpat import model
import numpy as np
import PIL.Image
import requests
import torch
from biological_coarse_graining.coarse_grain import sample_positions_from_image

def looks_like_vitruvian(world, cfg, level = 1, threshold=0.1):
  """
  Fitness that encourages the current particle positions to match an image silhouette.
  Uses a bidirectional Chamfer distance between particle positions and image mask points.

  Provide either `image` (URL or np.ndarray) or `emoji` (single character) to load a target.
  Returns negative Chamfer (higher is better when shapes match).
  """
  if world.x is None:
    return float(-1e6)
  # Load target points
  
  target_pts_np = sample_positions_from_image(level)

  if target_pts_np.shape[0] == 0:
    print("no target pts")
    return float(-1e6)  # harsh penalty if no target points

  target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)

  pts = torch.tensor(world.x.detach().cpu().numpy(), dtype=torch.float32, device=cfg.device)
  if pts.shape[0] == 0:
    return float(-1e6)

  dists = torch.cdist(pts, target_pts)  # (Na,Nb)
  # min_a_to_b = dists.min(dim=1).values.mean()
  # min_b_to_a = dists.min(dim=0).values.mean()
  # chamfer = (min_a_to_b + min_b_to_a).item()


  # # cardinality penalty
  # alpha = 1.
  # n_pred = pts.shape[0]
  # n_target = target_pts.shape[0]
  # cardinality_penalty = (abs(n_pred - n_target)) * alpha  # alpha ~ 1e-3 → 1e-1


  # # # overlap penalty
  # nn = torch.cdist(pts, pts) + torch.eye(pts.shape[0], device=pts.device) * 1e6
  # min_nn = nn.min(dim=1).values
  # overlap_penalty = 0.5 * torch.mean(torch.exp(-min_nn / 0.05))  # tune scale


  # Coverage: fraction of target points with any source closer than eps
  target_covered = (dists.min(dim=0).values < threshold).float().mean()#/target_pts.shape[0]  # in [0,1]
  cells_covered = (dists.min(dim=1).values < threshold).float().sum()/cfg.max_cells  # in [0,1]

  # fitness = - chamfer + ( target_covered + cells_covered) #+ overlap_penalty #+ cardinality_penalty
  # fitness = target_covered + cells_covered
  fitness = target_covered  + cells_covered #- overlap_penalty*0.001
  return float(fitness)


def separation_fitness(world, cfg, min_dist: float = 0.1) -> float:
  """
  Fitness encouraging particles to be at least `min_dist` away from all others.

  Uses the nearest-neighbor distance per particle. If the nearest neighbor distance
  is >= `min_dist`, then all neighbor distances are >= `min_dist` as well.

  Returns negative mean shortfall below `min_dist` (i.e., -mean(ReLU(min_dist - nn_dist))).
  Higher is better; best possible is 0 when all nn_dist >= min_dist.
  """
  if world.x is None:
    return float(-1e6)
  if world.x.shape[0] <= 1:
    # single particle trivially satisfies separation
    return float(0.0)

  pos = world.x  # (N,2) torch.Tensor on cfg.device
  d = torch.cdist(pos, pos)  # (N,N)
  N = d.shape[0]
  # ignore self-distances
  d = d + torch.eye(N, device=pos.device) * 1e6
  nn = d.min(dim=1).values  # nearest neighbor distance per particle
  shortfall = torch.relu(min_dist - nn)
  fitness = -shortfall.mean().item()
  return float(fitness)