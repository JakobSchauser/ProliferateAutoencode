"""
Simple utilities for sampling (x, y) positions from RGBA mask images.

Assumptions
- Pixels are either transparent, black, or white.
- Transparent pixels are ignored.
- Black pixels map to color -1, white pixels map to color +1.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from matplotlib import image as mpimg
from pathlib import Path
import matplotlib.pyplot as plt

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
	0: 128*64,
	1: 128*64,
	2: 256,
	3: 512,
	4: 1024,
}

def _to_float01(arr: np.ndarray) -> np.ndarray:
	"""Convert image array to float32 in [0, 1]."""
	arr = np.asarray(arr)
	if np.issubdtype(arr.dtype, np.floating):
		return arr.astype(np.float32)
	max_val = float(np.iinfo(arr.dtype).max)
	return arr.astype(np.float32) / max_val


def count_points_for_corse_graining(corse_graining: int, points_map: dict[int, int] | None = None) -> int:
	"""Return the number of points to sample for a given coarse-graining level."""
	if points_map is None:
		points_map = _DEFAULT_POINTS_MAP
	if corse_graining not in points_map:
		raise ValueError("corse_graining must be an integer present in points_map")
	return int(points_map[corse_graining])


def _classify_pixels(
	arr: np.ndarray,
	*,
	rgb_threshold: float = 0.5,
	alpha_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Return coordinates (row, col) of opaque pixels and their colors (-1 for black, +1 for white).

	Transparent pixels (alpha < alpha_threshold) are ignored. Black/white is determined using
	max-channel intensity against `rgb_threshold`.
	"""
	if arr.ndim == 2:
		rgb = np.stack([arr, arr, arr], axis=-1)
		alpha = np.ones(arr.shape, dtype=arr.dtype)
	elif arr.ndim == 3 and arr.shape[-1] == 3:
		rgb = arr
		alpha = np.ones(arr.shape[:2], dtype=arr.dtype)
	elif arr.ndim == 3 and arr.shape[-1] == 4:
		rgb = arr[..., :3]
		alpha = arr[..., 3]
	else:
		raise ValueError(f"Unsupported image array shape: {arr.shape}")

	opaque = alpha >= alpha_threshold
	luminance = rgb.max(axis=-1)
	black_mask = (luminance <= rgb_threshold) & opaque
	white_mask = (luminance > rgb_threshold) & opaque

	coords_black = np.argwhere(black_mask)
	coords_white = np.argwhere(white_mask)
	coords = np.concatenate([coords_black, coords_white], axis=0)
	colors = np.concatenate([
		np.full(coords_black.shape[0], -1.0, dtype=np.float32),
		np.full(coords_white.shape[0], 1.0, dtype=np.float32),
	], axis=0)
	return coords, colors


def _sample_grid(
	arr: np.ndarray,
	*,
	target_count: int,
	rgb_threshold: float = 0.5,
	alpha_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
	"""Sample approximately `target_count` points on an even grid over the image.

	Keeps only opaque pixels; assigns color -1 (black) or +1 (white) per grid location.
	Deterministic (no randomness).
	Returns positions as integer pixel coords in (row, col), and colors.
	"""
	h, w = arr.shape[0], arr.shape[1]

	# Choose grid resolution so that grid points ~ target_count over full image
	n_side = max(1, int(np.ceil(np.sqrt(max(1, target_count)))))
	ys = np.linspace(0, h - 1, num=n_side)
	xs = np.linspace(0, w - 1, num=n_side)

	# Prepare channels
	if arr.ndim == 2:
		rgb = np.stack([arr, arr, arr], axis=-1)
		alpha = np.ones(arr.shape, dtype=arr.dtype)
	elif arr.ndim == 3 and arr.shape[-1] == 3:
		rgb = arr
		alpha = np.ones(arr.shape[:2], dtype=arr.dtype)
	elif arr.ndim == 3 and arr.shape[-1] == 4:
		rgb = arr[..., :3]
		alpha = arr[..., 3]
	else:
		raise ValueError(f"Unsupported image array shape: {arr.shape}")

	opaque = alpha >= alpha_threshold
	luminance = rgb.max(axis=-1)

	coords_rc: list[tuple[int, int]] = []
	colors: list[float] = []

	for y in ys:
		r = int(round(y))
		for x in xs:
			c = int(round(x))
			if not opaque[r, c]:
				continue
			col = -1.0 if luminance[r, c] <= rgb_threshold else 1.0
			coords_rc.append((r, c))
			colors.append(col)

	if len(coords_rc) == 0:
		return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

	coords_rc_arr = np.asarray(coords_rc, dtype=np.int32)
	colors_arr = np.asarray(colors, dtype=np.float32)
	return coords_rc_arr, colors_arr


def sample_positions_from_image(
	corse_graining: int,
	*,
	points_map: dict[int, int] | None = None,
	rgb_threshold: float = 0.5,
	alpha_threshold: float = 0.5,
	seed: int | None = None,
	return_types: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
	"""
	Return sampled positions (and optional colors) from an RGBA mask image.

	- Black pixels -> color -1
	- White pixels -> color +1
	- Transparent pixels are ignored
	- Positions are normalized the same way as the previous implementation
	"""
	if corse_graining < 0 or corse_graining >= len(image_paths):
		raise ValueError(f"corse_graining must be between 0 and {len(image_paths) - 1}")

	arr = _to_float01(mpimg.imread(image_paths[corse_graining]))
	target = count_points_for_corse_graining(corse_graining, points_map)
	coords_rc, colors = _sample_grid(
		arr,
		target_count=target,
		rgb_threshold=rgb_threshold,
		alpha_threshold=alpha_threshold,
	)

	if coords_rc.shape[0] == 0:
		return (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)) if return_types else np.empty((0, 2), dtype=np.float32)

	# Convert (row, col) -> (x, y)
	positions = coords_rc[:, [1, 0]].astype(np.float32)

	# Keep legacy normalization expected by downstream code
	positions[:, 0] -= 450.0
	positions[:, 1] -= 150.0
	positions /= 500.0

	if return_types:
		return positions, colors.astype(np.float32)
	return positions


__all__ = [
	"count_points_for_corse_graining",
	"sample_positions_from_image",
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

