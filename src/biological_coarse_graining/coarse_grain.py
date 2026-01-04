"""
Utilities for sampling pixel positions from images based on
black/transparent regions, with a simple coarse graining control.

This module avoids extra dependencies by using matplotlib + numpy
to read images and compute masks.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

levels = [0, 1, 2, 3, 4]

here = Path(__file__).resolve().parent

outdir = here / "results"

# Collect four variant masks
image_paths = [
	str(here / "Vitruvian_0.png"),
	str(here / "Vitruvian_1.png"),
	str(here / "Vitruvian_2.png"),
	str(here / "Vitruvian_3.png"),
	str(here / "Vitruvian_4.png"),
]

# Default number of points to return for each coarse-graining level (1-4).
_DEFAULT_POINTS_MAP = {
	0: 128,
	1: 128,
	2: 256,
	3: 512,
	4: 1024,
}

# Sampling mode: "random" (default) or "grid".
# "random" samples uniformly from valid mask pixels without replacement.
# "grid" lays a roughly uniform grid over the image and keeps points whose
# grid location falls on valid mask pixels; if not enough points, it fills
# the remainder randomly from the mask.
MODE: str = "grid"


def _opaque_black_mask(arr: np.ndarray, rgb_threshold: float) -> np.ndarray:
	"""
	Build a boolean mask of pixels that are "black" AND opaque (if an alpha channel exists).

	Supports grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4) arrays.
	Thresholds should be given in the data scale of `arr` (0-1 for float, 0-255 for uint8/uint16).
	"""
	if arr.ndim == 2:
		# Grayscale: black if intensity below threshold.
		return arr <= rgb_threshold

	if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
		raise ValueError(f"Unsupported image array shape: {arr.shape}")

	rgb = arr[..., :3]
	# Define black as any pixel whose max channel is below threshold.
	black_mask = (rgb.max(axis=-1) <= rgb_threshold)

	if arr.shape[-1] == 4:
		alpha = arr[..., 3]
		# Consider pixels opaque if alpha >= 0.5 for float images, or >= 128 for integer images.
		if np.issubdtype(arr.dtype, np.floating):
			opaque_mask = alpha >= 0.5
		else:
			opaque_mask = alpha >= 128
		return black_mask & opaque_mask

	return black_mask


def _normalize_thresholds(arr: np.ndarray, rgb_threshold: float | int = 10, alpha_threshold: float | int = 10) -> Tuple[float, float]:
	"""
	Convert thresholds to the dtype/range of the incoming array.
	For float images (0..1), thresholds are scaled to [0,1]. For integer (e.g., uint8 0..255), keep as-is.
	"""
	if np.issubdtype(arr.dtype, np.floating):
		# Scale typical 0..255 integer thresholds down to 0..1 if the caller provided int-like values.
		rgb_thr = float(rgb_threshold)
		alpha_thr = float(alpha_threshold)
		# If thresholds look like integer ranges, normalize.
		if rgb_thr > 1.0 or alpha_thr > 1.0:
			rgb_thr /= 255.0
			alpha_thr /= 255.0
		return rgb_thr, alpha_thr
	else:
		return float(rgb_threshold), float(alpha_threshold)


def count_points_for_corse_graining(corse_graining: int, points_map: dict[int, int] | None = None) -> int:
	"""
	Return the number of points to sample for a given `corse_graining` level.

	Parameters
	- corse_graining: int in [1, 5]
	- points_map: optional override mapping for levels 1..5

	Raises
	- ValueError if `corse_graining` is outside 1..5 or not in the mapping.
	"""
	if points_map is None:
		points_map = _DEFAULT_POINTS_MAP
	if corse_graining not in points_map:
		raise ValueError("corse_graining must be an integer in [1, 5]")
	return int(points_map[corse_graining])


def _prepare_mask_and_coords(
	image_path: str | Sequence[str],
	rgb_threshold: float | int,
	alpha_threshold: float | int,
):
	"""
	Load one or more images and build a combined opaque-black mask.
	If multiple images are provided, they must share the same dimensions;
	masks are OR-combined.
	Returns the first image array for size reference, the combined mask,
	and its coordinates.
	"""
	if isinstance(image_path, (list, tuple)):
		paths = list(image_path)
		if not paths:
			raise ValueError("image_path list is empty")
		base_arr = mpimg.imread(paths[0])
		rgb_thr, _ = _normalize_thresholds(base_arr, rgb_threshold, alpha_threshold)
		mask = _opaque_black_mask(base_arr, rgb_thr)
		h, w = base_arr.shape[0], base_arr.shape[1]
		for p in paths[1:]:
			arr_i = mpimg.imread(p)
			if arr_i.shape[0] != h or arr_i.shape[1] != w:
				raise ValueError("All images must have identical dimensions for mask combination")
			rgb_thr_i, _ = _normalize_thresholds(arr_i, rgb_threshold, alpha_threshold)
			mask |= _opaque_black_mask(arr_i, rgb_thr_i)
		coords_rc = np.argwhere(mask)
		return base_arr, mask, coords_rc
	else:
		arr = mpimg.imread(image_path)
		rgb_thr, _ = _normalize_thresholds(arr, rgb_threshold, alpha_threshold)
		mask = _opaque_black_mask(arr, rgb_thr)
		coords_rc = np.argwhere(mask)
		return arr, mask, coords_rc


def sample_positions_random_from_image(
	image_path: str | Sequence[str],
	corse_graining: int,
	*,
	points_map: dict[int, int] | None = None,
	rgb_threshold: float | int = 10,
	alpha_threshold: float | int = 10,
	seed: int | None = None,
) -> List[Tuple[int, int]]:
	arr, mask, coords_rc = _prepare_mask_and_coords(image_path, rgb_threshold, alpha_threshold)
	if coords_rc.size == 0:
		return []
	total = min(count_points_for_corse_graining(corse_graining, points_map), coords_rc.shape[0])
	rng = np.random.default_rng(seed)
	idx = rng.choice(coords_rc.shape[0], size=total, replace=False)
	sampled = coords_rc[idx]
	return [(int(c), int(r)) for r, c in sampled]


def sample_positions_grid_from_image(
	image_path: str | Sequence[str],
	corse_graining: int,
	*,
	points_map: dict[int, int] | None = None,
	rgb_threshold: float | int = 10,
	alpha_threshold: float | int = 10,
	seed: int | None = None,
) -> List[Tuple[int, int]]:
	# Grid mode: place a grid and sample points where the grid hits valid mask pixels.
	# Fineness doubles with each level; we do not enforce an exact count.
	arr, mask, _ = _prepare_mask_and_coords(image_path, rgb_threshold, alpha_threshold)
	h, w = arr.shape[0], arr.shape[1]

	# Base grid resolution; start modestly to avoid zero-cell sizes.
	base = 16*2*2
	n_side = max(1, int(base * np.sqrt(int(2** max(0, int(corse_graining) - 1)))))

	xs = np.linspace(0, w - 1, num=n_side)
	ys = np.linspace(0, h - 1, num=n_side)

	points: List[Tuple[int, int]] = []
	for y in ys:
		r = int(round(y))
		for x in xs:
			c = int(round(x))
			if mask[r, c]:
				points.append((c, r))

	# Deduplicate in case rounding collides; preserve order
	seen = set()
	dedup: List[Tuple[int, int]] = []
	for p in points:
		if p not in seen:
			seen.add(p)
			dedup.append(p)

	return dedup


def sample_positions_from_image(
	corse_graining: int,
	*,
	points_map: dict[int, int] | None = None,
	rgb_threshold: float | int = 10,
	alpha_threshold: float | int = 10,
	seed: int | None = None,
	return_types: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
	"""
	Return sampled positions as a NumPy array of shape (N, 2).

	Dispatches to either grid or random sampling based on the global `MODE`.
	The returned array has integer dtype and columns ordered as (x, y).
	"""

	image_path = image_paths[corse_graining],

	mode = MODE.lower()
	if mode == "grid":
		positions_list = sample_positions_grid_from_image(
			image_path,
			corse_graining,
			points_map=points_map,
			rgb_threshold=rgb_threshold,
			alpha_threshold=alpha_threshold,
			seed=seed,
		)
	else:
		positions_list = sample_positions_random_from_image(
			image_path,
			corse_graining,
			points_map=points_map,
			rgb_threshold=rgb_threshold,
			alpha_threshold=alpha_threshold,
			seed=seed,
		)

	positions = np.asarray(positions_list, dtype=np.int32)

	# "normalize"
	positions[:,0] -= 450
	positions[:,1] -= 225

	positions = positions / 500.

	# target_color = np.where(positions[:,0] > 0., 1., -1.)
	
	cxx = positions[:,0]
	target_color = (cxx - cxx.min())/(cxx.max() - cxx.min())
	if positions.size == 0:
		return np.empty((0, 2), dtype=np.int32)
	if positions.ndim == 1:
		# Ensure shape (N, 2) even for single point
		positions = positions.reshape(-1, 2)
	if return_types:
		return positions, target_color
	
	return positions


__all__ = [
	"count_points_for_corse_graining",
	"sample_positions_from_image",
	"sample_positions_random_from_image",
	"sample_positions_grid_from_image",
]



def main(argv: Sequence[str] | None = None) -> None:
	"""
	Generate scatter examples into the results/ folder using `plt.scatter`.

	If `--image` is provided, sample positions from that image for
	coarse_graining levels 1..4.
	"""

	# Read image once to get dimensions for plotting limits.

	# Ensure output directory exists
	outdir.mkdir(parents=True, exist_ok=True)

	for level in levels:
		positions = sample_positions_from_image(
			level,
			seed=42,
		)

		xs = positions[:, 0]
		ys = positions[:, 1]

		plt.figure(figsize=(5, 5), dpi=150)
		plt.scatter(xs, ys, s=6, c="tab:red", alpha=0.75, edgecolors="none")
		plt.title(f"coarse_graining={level}, points={len(positions)}")
		plt.gca().set_aspect("equal", adjustable="box")
		plt.tight_layout()
		out_file = outdir / f"scatter_cg{level}.png"
		plt.savefig(out_file)
		plt.close()

	print(f"Saved scatter examples to: {outdir}")


if __name__ == "__main__":
	main()

