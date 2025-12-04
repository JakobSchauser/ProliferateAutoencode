import io
import os
from pyexpat import model
import numpy as np
import PIL.Image
import requests
import torch



def load_image(url, max_size=256/2):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
  img = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img


def load_emoji(emoji):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url)

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
  min_a_to_b = dists.min(dim=1).values.mean()
  min_b_to_a = dists.min(dim=0).values.mean()
  chamfer = (min_a_to_b + min_b_to_a).item()

  fitness = -chamfer
  return float(fitness)