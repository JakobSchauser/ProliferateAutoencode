import io
import os
from pyexpat import model
import numpy as np
import PIL.Image
import requests
import torch
import matplotlib.image as mpimg
from biological_coarse_graining.coarse_grain import sample_positions_from_image, image_paths


def _debug_check(name: str, t: torch.Tensor):
  if os.environ.get("DEBUG_NANS"):
    if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
      nf = (~torch.isfinite(t)).sum().item()
      print(f"[NaN][fitness] {name} non-finite: count={nf}, shape={tuple(t.shape)}")


def _safe_div(numer: torch.Tensor, denom: torch.Tensor | float, eps: float = 1e-12) -> torch.Tensor:
  if not isinstance(denom, torch.Tensor):
    denom_t = torch.tensor(float(denom), dtype=numer.dtype, device=numer.device)
  else:
    denom_t = denom
  return numer / (denom_t + eps)

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
  _debug_check("looks_like_vitruvian.dists", dists)
  dists = torch.nan_to_num(dists)

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
  _debug_check("looks_like_vitruvian_gaussian.d2", d2)
  d2 = torch.nan_to_num(d2)
  sigma2 = max(1e-12, float(gauss_width) ** 2)
  gauss = torch.exp(- d2 / (2.0 * sigma2))
  _debug_check("looks_like_vitruvian_gaussian.gauss", gauss)
  field_max = gauss.max(dim=1).values  # (Nb,)
  covered_target = (field_max >= float(threshold)).float().mean()

  field_max_ct = gauss.max(dim=0).values  # (Na,)
  covered_cells = (field_max_ct >= float(threshold)).float().mean()

  w = 1./(1.+4.)
  return (float(covered_target) * float(covered_cells))#*w


def rotate_positions(positions, angle_rad):
  """
  Rotate 2D positions by angle_rad.

  positions: (N,2) torch.Tensor
  angle_rad: float
  returns: (N,2) torch.Tensor
  """
  # Support both torch.Tensor and numpy.ndarray inputs
  if isinstance(positions, np.ndarray):
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return positions @ R.T
  else:
    c = torch.cos(torch.tensor(angle_rad, device=positions.device))
    s = torch.sin(torch.tensor(angle_rad, device=positions.device))
    R = torch.tensor([[c, -s], [s, c]], device=positions.device)  # (2,2)
    rotated = positions @ R.T  # (N,2)
    return rotated

def looks_like_vitruvian_gaussian_rotated(
    world,
    cfg,
    level: int = 1,
    gauss_width: float = 0.1,
    threshold: float = 0.2,
    angle_rad: float = 0.0,
) -> float:
  """
  Like looks_like_vitruvian_gaussian, but rotates the cell positions by angle_rad before evaluation.
  """

  if world.x is None:
    assert False, "world.x is None in looks_like_vitruvian_gaussian"

  target_pts_np = sample_positions_from_image(level)
  if target_pts_np.shape[0] == 0:
    print("no target pts")
    assert False, "no target points in looks_like_vitruvian_gaussian"

  target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)
  
  # Rotate using per-run angle provided to the function
  target_pts = rotate_positions(target_pts, float(angle_rad))
  
  pts = world.x
  if pts.shape[0] == 0:
    return float(0.0)

  d2 = torch.cdist(target_pts, pts) ** 2  # (Nb, Na)
  _debug_check("looks_like_vitruvian_gaussian_rotated.d2", d2)
  d2 = torch.nan_to_num(d2)
  sigma2 = max(1e-12, float(gauss_width) ** 2)
  gauss = torch.exp(- d2 / (2.0 * sigma2))
  _debug_check("looks_like_vitruvian_gaussian_rotated.gauss", gauss)
  field_max = gauss.max(dim=1).values  # (Nb,)
  covered_target = (field_max >= float(threshold)).float().mean()

  field_max_ct = gauss.max(dim=0).values  # (Na,)
  covered_cells = (field_max_ct >= float(threshold)).float().mean()

  w = 1./(1.+4.)


  # keep a minimum distance between cells
  closest_dist = torch.cdist(pts, pts)  # (N,N)
  closest_dist = torch.nan_to_num(closest_dist)

  closest_dist = closest_dist + torch.eye(closest_dist.shape[0], device=pts.device) * 1e6
  min_dist = closest_dist.min(dim=1).values  # (N,)
  separation_shortfall = torch.relu(0.05 - min_dist)  # (N,)
  separation_penalty = separation_shortfall.mean()

  return (float(covered_target) * float(covered_cells) ) #- float(separation_penalty)*0.1


def color_fitness(world, cfg) -> float:
  """
  Color-only fitness: evaluates how well the cell colors match target colors at nearest target points.
  Target colors are derived from the source image: red-dominant => +1, blue-dominant => -1.
  Each target point is matched to its nearest cell; correctness is the mean sign match of world.mol[:, -1].
  """
  if world.x is None:
    assert False, "world.x is None in color_fitness"

  target_pts_np, target_color = sample_positions_from_image(1, return_types = True)
  if target_pts_np.shape[0] == 0:
    print("no target pts")
    assert False, "no target points in color_fitness"

  target_color = torch.tensor(target_color, device=cfg.device, dtype=torch.float32)

  target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)

  pts = world.x
  if pts.shape[0] == 0:
    return float(0.0)

  d2 = torch.cdist(target_pts, pts) ** 2
  _debug_check("color_fitness.d2", d2)
  d2 = torch.nan_to_num(d2)

  # Color correctness: nearest-cell assignment per target
  nn_idx = d2.argmin(dim=1)
  pred_color = world.mol[nn_idx, -1]

  color_correct = 1 - (pred_color - target_color).abs().float().mean()
  return float(color_correct)


def color_fitness_rotated(world, cfg, angle_rad: float = 0.0) -> float:
  """
  Color-only fitness with rotation: evaluates how well the cell colors match target colors at nearest target points.
  Target colors are derived from the source image: red-dominant => +1, blue-dominant => -1.
  Each target point is matched to its nearest cell; correctness is the mean sign match of world.mol[:, -1].
  The target points are rotated by angle_rad before evaluation.
  """
  if world.x is None:
    assert False, "world.x is None in color_fitness_rotated"

  target_pts_np, target_color = sample_positions_from_image(1, return_types = True)

  # target_color = torch.tensor(target_color, device=cfg.device, dtype=torch.float32)
  # target_color = torch.ones_like(target_color)  # all +1 for testing
  
  
  target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=cfg.device)
  target_pts = rotate_positions(target_pts, float(angle_rad))

  pts = world.x
  if pts.shape[0] == 0:
    return float(0.0)

  d2 = torch.cdist(target_pts, pts) ** 2
  _debug_check("color_fitness_rotated.d2", d2)
  d2 = torch.nan_to_num(d2)

  # Color correctness: nearest-cell assignment per target
  nn_idx = d2.argmin(dim=0)

  # penalty = (d2.min(dim=0).values > 0.01).float().sum()

  pred_color = world.mol[:, -1].detach()
  targets = target_color[nn_idx]
  # targets = world.mol[:, 0].detach()
  # print("+", world.mol[:10, 0])
  # print("-",world.mol[:10, -1])

  color_correct = 1 - (pred_color - targets).abs().float().mean()
  return float(color_correct)


def color_and_cover(world, cfg, angle_rad: float = 0.0, level: int = 1,
                    gauss_width: float = 0.1, threshold: float = 0.85) -> float:
  """
  Combined color and coverage fitness.
  """
  color = color_fitness_rotated(
      world,
      cfg,
      angle_rad=angle_rad,
  )

  coverage = looks_like_vitruvian_gaussian_rotated(
      world,
      cfg,
      level=level,
      gauss_width=gauss_width,
      threshold=threshold,
      angle_rad=angle_rad,
  )
  return float(coverage * color)


def get_all_cells_to_origin_distance(world, cfg) -> float:
  """
  Returns the mean distance of all cells to the origin (0,0).
  """
  if world.x is None:
    return float(1e6)
  pos = world.x  # (N,2) torch.Tensor
  dists = torch.norm(pos, dim=1)  # (N,)
  mean_dist = dists.mean().item()
  return 1 - float(mean_dist)


def uniqueness_of_molecules(world, cfg) -> float:
  """
  Returns a fitness score based on the uniqueness of molecule vectors among cells.
  Higher score for more unique molecule vectors.
  """
  if world.mol is None or world.mol.shape[0] == 0:
    return float(0.0)
  
  mols = world.mol  # (N, C) torch.Tensor
  N = mols.shape[0]
  
  # Compute pairwise distances between molecule vectors
  dists = torch.cdist(mols, mols)  # (N, N)
  _debug_check("uniqueness_of_molecules.dists", dists)
  dists = torch.nan_to_num(dists)
  
  

  # Count number of unique molecule vectors based on a distance threshold
  if N < 2:
    return float(0.0)
  threshold = 1.
  unique_mask = (dists > threshold).float()
  denom = float(N * (N - 1))
  score = _safe_div(unique_mask.sum(), denom)
  _debug_check("uniqueness_of_molecules.score", score)
  return float(score) # goes from 0 to 1 where 1 is all unique


def uniqueness_of_final_n_molecules(world, cfg, n: int = -1) -> float:
  """
  Returns a fitness score based on the uniqueness of the nth molecule channels among cells.
  Higher score for more unique values in the nth molecule channels.
  """
  if world.mol is None or world.mol.shape[0] == 0:
    return float(0.0)
  
  mols = world.mol[:, n:]  # (N, n) torch.Tensor
  N = mols.shape[0]
  
  # Compute pairwise distances between nth molecule vectors
  dists = torch.cdist(mols, mols)  # (N, N)
  _debug_check("uniqueness_of_final_n_molecules.dists", dists)
  dists = torch.nan_to_num(dists)

  # only take upper triangle without diagonal
  dists = dists.triu(diagonal=1)
 
  if N < 2:
    return float(0.0)
  N_entries = N * (N - 1) / 2
  # Count number of unique nth molecule vectors based on a distance threshold
  threshold = 0.3
  unique_mask = (dists > threshold).float()
  score = _safe_div(unique_mask.sum(), float(N_entries))
  _debug_check("uniqueness_of_final_n_molecules.score", score)
  return float(score) # goes from 0 to 1 where 1 is all unique


def local_agreement_global_uniqueness(world, cfg) -> float:
  """
  Returns a fitness score that combines local agreement and global uniqueness of molecule vectors.
  Higher score for high local agreement and high global uniqueness.
  """
  if world.mol is None or world.mol.shape[0] == 0:
    return float(0.0)
  
  mols = world.mol  # (N, C) torch.Tensor
  N = mols.shape[0]
  
  # Compute pairwise distances between molecule vectors
  moleculardists = torch.cdist(mols, mols)  # (N, N)
  spatialdists = torch.cdist(world.x, world.x)  # (N, N)
  moleculardists = torch.nan_to_num(moleculardists)
  spatialdists = torch.nan_to_num(spatialdists)

  # Local agreement: average distance of molecule vectors for nearby cells
  local_threshold = 0.1
  local_mask = (spatialdists < local_threshold).float()
  local_agreement = (moleculardists * local_mask).sum() / (local_mask.sum() + 1e-6)

  # Global uniqueness: average distance of molecule vectors for all cell pairs
  if N < 2:
    return float(0.0)
  global_uniqueness = _safe_div(moleculardists.sum(), float(N * (N - 1)))

  # Combine local agreement and global uniqueness into a single fitness score
  fitness = (1.0 / (1.0 + local_agreement)) * global_uniqueness
  _debug_check("local_agreement_global_uniqueness.fitness", fitness)
  return float(fitness)