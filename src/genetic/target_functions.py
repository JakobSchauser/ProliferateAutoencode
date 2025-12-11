import io
import os
from pyexpat import model
import numpy as np
import PIL.Image
import requests
import torch



def load_image(url, max_size=256/8):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
  img = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img


def load_emoji(emoji):
  # code = hex(ord(emoji))[2:].lower()
  # url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code

  # print(code)
  # return load_image(url)

  # for testing
  localpath = "lizard.png"
  if os.path.exists(localpath):
    max_size=256/8
    img = PIL.Image.open(localpath)
    img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
    img = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    img[..., :3] *= img[..., 3:]
    return img


def from_image_to_postions(img, threshold=0.1):
  # img: np.ndarray HxWx4 (RGBA premultiplied) from load_image
  # Return Nx2 coordinates in [-1,1] normalized image space where alpha > threshold
  h, w = img.shape[:2]
  alpha = img[..., 3]
  mask = alpha > threshold
  ys, xs = np.where(mask)
  if xs.size == 0:
    return np.zeros((0, 2), dtype=np.float32)
  # Normalize to [-1,1] with origin at image center
  x_norm = (xs / (w - 1)) * 2.0 - 1.0
  y_norm = (ys / (h - 1)) * 2.0 - 1.0
  # Flip y to match Cartesian coordinates (optional)
  y_norm = -y_norm
  pts = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)
  return pts


def correct_cell_count_fitness(world, cfg) -> float:
  if world.x is None:
    return float(-1e6)
  curr = world.x.shape[0]
  fitness = -((curr - cfg.target_count) ** 2)
  return float(fitness)


def looks_like_image_fitness(world, cfg, image=None, emoji=None, threshold=0.1):
  """
  Fitness that encourages the current particle positions to match an image silhouette.
  Uses a bidirectional Chamfer distance between particle positions and image mask points.

  Provide either `image` (URL or np.ndarray) or `emoji` (single character) to load a target.
  Returns negative Chamfer (higher is better when shapes match).
  """
  if world.x is None:
    return float(-1e6)
  # Load target points
  if image is not None:
    if isinstance(image, str):
      img = load_image(image)
    else:
      img = np.asarray(image, dtype=np.float32)
  elif emoji is not None:
    img = load_emoji(emoji)
  else:
    raise ValueError("Provide either image (url/array) or emoji")

  target_pts_np = from_image_to_postions(img, threshold=threshold)
  if target_pts_np.shape[0] == 0:
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


  # # overlap penalty
  # nn = torch.cdist(pts, pts) + torch.eye(pts.shape[0], device=pts.device) * 1e6
  # min_nn = nn.min(dim=1).values
  # overlap_penalty = 0.0
  # if pts.shape[0] > 1:
  #     overlap_penalty = 0.5 * torch.mean(torch.exp(-min_nn / 0.01))  # tune scale


    # Coverage: fraction of target points with any source closer than eps
  eps = 0.05  # tune: how close a cell must be to count as "covering" a target point
  target_covered = (dists.min(dim=0).values < eps).float().mean()  # in [0,1]
  cells_covered = (dists.min(dim=1).values < eps).float().mean()  # in [0,1]

  # fitness = - chamfer + ( target_covered + cells_covered) #+ overlap_penalty #+ cardinality_penalty
  # fitness = target_covered + cells_covered
  fitness = target_covered 
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