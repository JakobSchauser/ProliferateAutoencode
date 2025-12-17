import io
import os
from pyexpat import model
import numpy as np
import PIL.Image
import requests
import torch
from biological_coarse_graining.coarse_grain import sample_positions_from_image

def looks_like_vitruvian(world, cfg, level = 1, threshold=0.1):
  if world.x is None:
    assert False, "world.x is None in looks_like_vitruvian"
  
  target_pts_np = sample_positions_from_image(level)

  if target_pts_np.shape[0] == 0:
    print("no target pts")
    assert False, "no target points in looks_like_vitruvian"

  target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)

  pts = torch.tensor(world.x.detach().cpu().numpy(), dtype=torch.float32, device=cfg.device)

  dists = torch.cdist(pts, target_pts)  # (Na,Nb)

  # Coverage: fraction of target points with any source closer than eps
  target_covered = (dists.min(dim=0).values < threshold).float().mean()#/target_pts.shape[0]  # in [0,1]
  cells_covered = (dists.min(dim=1).values < threshold).float().sum()/target_pts.shape[0]  # in [0,1]

  cell_cover_gaussian_score = torch.exp(- (dists **2) / (2*(threshold**2)) ).sum(dim=1)
  
  cells_covered = min(cells_covered, 1.0)

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


def looks_like_vitruvian_gaussian(world, cfg, level: int = 1, gauss_width: float = 0.1, threshold: float = 0.2) -> float:
  """
  Gaussian max-coverage fitness:
  - Treat each cell position as a Gaussian bump with width `gauss_width`.
  - For each target point, evaluate the field as the max over all Gaussians.
  - Score is the fraction of target points with field >= `threshold`.

  Returns a value in [0,1].
  """
  if world.x is None:
    assert False, "world.x is None in looks_like_vitruvian_gaussian"

  target_pts_np = sample_positions_from_image(level)
  if target_pts_np.shape[0] == 0:
    print("no target pts")
    assert False, "no target points in looks_like_vitruvian_gaussian"

  target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)
  pts = world.x
  if pts.shape[0] == 0:
    return float(0.0)

  d2 = torch.cdist(target_pts, pts) ** 2  # (Nb, Na)
  sigma2 = max(1e-12, float(gauss_width) ** 2)
  gauss = torch.exp(- d2 / (2.0 * sigma2))
  field_max = gauss.max(dim=1).values  # (Nb,)
  covered_target = (field_max >= float(threshold)).float().mean()

  field_max_ct = gauss.max(dim=0).values  # (Na,)
  covered_cells = (field_max_ct >= float(threshold)).float().mean()

  return (float(covered_target) + float(covered_cells))*0.5
