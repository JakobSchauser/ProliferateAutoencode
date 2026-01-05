import math
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Voronoi


class ParticleNCA(nn.Module):
	"""
	Off-grid (particle-based) Neural Cellular Automata in PyTorch.

	Each particle (cell) has features:
	  - x, y: position
	  - angle: orientation in radians
	  - molecules: hidden channels of size `molecule_dim`

	Neighborhood: k-NN with cutoff radius. Messages are computed using
	relative features (dx, dy, d_angle, neighbor molecules - self molecules).

	The model predicts deltas for updates:
	  (dx, dy, d_angle, d_molecules)

	Usage:
	  nca = ParticleNCA(molecule_dim=16, k=16, cutoff=0.25)
	  x = torch.randn(N, 2)             # positions
	  angle = torch.randn(N, 1)         # angles (radians)
	  mol = torch.randn(N, 16)          # molecules
	  gen = torch.zeros(N, 1)           # generation number (int-like float)
	  dx, dtheta, dmol, divide_logit = nca(x, angle, mol, gen)
	"""

	def __init__(
		self,
		molecule_dim: int,
		k: int = 16,
		cutoff: float = 0.25,
		message_hidden: int = 64,
		update_hidden: int = 64,
		aggregate: str = "sum",  # or "mean"
		positional_encoding: bool = True,
		angle_sin_cos: bool = True,
	):
		super().__init__()
		self.molecule_dim = molecule_dim
		self.k = k
		self.cutoff = cutoff
		self.aggregate = aggregate
		self.positional_encoding = positional_encoding
		self.angle_sin_cos = angle_sin_cos

		# Message MLP takes relative inputs
		# Inputs per edge:
		# - dx, dy, r, optionally PE for r
		# - sin(d_angle), cos(d_angle) or raw d_angle
		# - (mol_j - mol_i) molecules difference
		rel_geom_dim = 3  # dx, dy, r
		if self.positional_encoding:
			rel_geom_dim += 4  # simple Fourier features for r

		angle_dim = 2 if self.angle_sin_cos else 1
		rel_input_dim = rel_geom_dim + angle_dim + molecule_dim + molecule_dim  # include mol_i

		self.message_mlp = nn.Sequential(
			nn.Linear(rel_input_dim, message_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(message_hidden, message_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(message_hidden, message_hidden),
			nn.ReLU(inplace=True),
		)

		# Update MLP maps aggregated message + self features to deltas
		# Self features fed to update head: angle repr + molecules + generation
		self_feat_dim = (2 if self.angle_sin_cos else 1) + molecule_dim + 1
		upd_in = message_hidden + self_feat_dim
		self.update_mlp = nn.Sequential(
			nn.Linear(upd_in, update_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(update_hidden, update_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(update_hidden, update_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(update_hidden, update_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(update_hidden, 2 + 1 + molecule_dim + 1),  # dx, dy, d_angle, d_molecules, divide_logit
		)

	@staticmethod
	def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
		# x: (N, 2)
		# returns (N, N) pairwise euclidean distances
		# Using cdist might be memory hungry; manual is fine for moderate N.
		# dist_ij = ||x_i - x_j||
		xi = x.unsqueeze(1)  # (N,1,2)
		xj = x.unsqueeze(0)  # (1,N,2)
		d = xi - xj
		return torch.sqrt(torch.clamp((d ** 2).sum(-1), min=1e-12))

	@staticmethod
	def _fourier_features(t: torch.Tensor, n_freqs: int = 2) -> torch.Tensor:
		# Simple Fourier features: [sin(2^k t), cos(2^k t)] for k=0..n_freqs-1
		outs = []
		for k in range(n_freqs):
			f = 2.0 ** k
			outs.append(torch.sin(f * t))
			outs.append(torch.cos(f * t))
		return torch.cat(outs, dim=-1)

	def _hard_cutoff_neighbors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		# Previous k-NN cutoff implementation kept for reference:
		with torch.no_grad():
			dist = self._pairwise_dist(x)
			N = dist.shape[0]
			dist = dist + torch.eye(N, device=x.device) * 1e6
			mask = dist <= self.cutoff
			dst, src = torch.where(mask)
		return src, dst
		with torch.no_grad():
			points = x.detach().cpu().numpy()
			N = points.shape[0]
			if N <= 1:
				return (
					torch.empty(0, dtype=torch.long, device=x.device),
					torch.empty(0, dtype=torch.long, device=x.device),
				)
			if N < 4:
				# Fallback to dense distance cutoff when Voronoi cannot be constructed
				dist = self._pairwise_dist(x)
				dist = dist + torch.eye(N, device=x.device) * 1e6
				mask = dist <= self.cutoff
				dst, src = torch.where(mask)
				return src, dst
			vor = Voronoi(points)
			pairs = vor.ridge_points  # (M,2) unique undirected edges
			if pairs.size == 0:
				return (
					torch.empty(0, dtype=torch.long, device=x.device),
					torch.empty(0, dtype=torch.long, device=x.device),
				)
			# limit neighbors by cutoff to avoid long-range connections
			vec = points[pairs[:, 0]] - points[pairs[:, 1]]
			if self.cutoff is not None:
				mask = (vec ** 2).sum(axis=1) <= float(self.cutoff*10.) ** 2
				pairs = pairs[mask]
				vec = vec[mask]
			if pairs.size == 0:
				return (
					torch.empty(0, dtype=torch.long, device=x.device),
					torch.empty(0, dtype=torch.long, device=x.device),
				)
			# add both directions for message passing
			pairs_bidirectional = np.vstack([pairs, pairs[:, ::-1]])
			pairs_unique = np.unique(pairs_bidirectional, axis=0)
			src = torch.from_numpy(pairs_unique[:, 1]).to(x.device, dtype=torch.long)
			dst = torch.from_numpy(pairs_unique[:, 0]).to(x.device, dtype=torch.long)
		return src, dst
	
	def forward(
		self,
		x: torch.Tensor,           # (N, 2)
		angle: torch.Tensor,       # (N, 1)
		molecules: torch.Tensor,   # (N, molecule_dim)
		generation: torch.Tensor,  # (N, 1) integer-like counter
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		N = x.shape[0]
		device = x.device

		src, dst = self._hard_cutoff_neighbors(x)  # (E,), (E,) where E = number of edges
		E = src.shape[0]

		# Build global edge list - already in the right format from _hard_cutoff_neighbors
		# src: neighbor indices j (E,)
		# dst: target indices i (E,)

		# Gather features per edge
		x_i = x[dst]        # (E,2)
		x_j = x[src]        # (E,2)
		angle_i = angle[dst]  # (E,1)
		angle_j = angle[src]  # (E,1)
		mol_i = molecules[dst]  # (E,C)
		mol_j = molecules[src]  # (E,C)

		# Relative features per edge
		dxdy = x_j - x_i                # (E,2)
		r = torch.sqrt(torch.clamp((dxdy ** 2).sum(-1, keepdim=True), min=1e-12))  # (E,1)
		d_angle = angle_j - angle_i     # (E,1)
		if self.angle_sin_cos:
			ang_feat = torch.cat([torch.sin(d_angle), torch.cos(d_angle)], dim=-1)  # (E,2)
		else:
			ang_feat = d_angle  # (E,1)
		d_mol = mol_j - mol_i           # (E,C)

		# Optional positional encoding on r (no generation as edge feature)
		if self.positional_encoding:
			r_pe = self._fourier_features(r)  # (E,4)
			rel_geom = torch.cat([dxdy, r, r_pe], dim=-1)  # (E, 2+1+4)
		else:
			rel_geom = torch.cat([dxdy, r], dim=-1)  # (E,3)

		# Per-edge input to message MLP
		rel_in = torch.cat([rel_geom, ang_feat, d_mol, mol_i], dim=-1)  # (E + molecule_dim + angle_dim + rel_geom_dim)

		msg = self.message_mlp(rel_in)  # (E,H)

		# Aggregate messages to nodes via index_add
		agg = torch.zeros(N, msg.shape[-1], device=device)
		agg.index_add_(0, dst, msg)
		if self.aggregate == "mean":
			denom = torch.zeros(N, 1, device=device)
			ones = torch.ones(E, 1, device=device)
			denom.index_add_(0, dst, ones)
			denom = torch.clamp(denom, min=1.0)
			agg = agg / denom
		elif self.aggregate != "sum":
			raise ValueError(f"Unknown aggregate: {self.aggregate}")

		# Self features for update head
		if self.angle_sin_cos:
			self_ang = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (N,2)
		else:
			self_ang = angle  # (N,1)
		self_feat = torch.cat([self_ang, molecules, generation], dim=-1)  # (N, 2/1 + C + 1)

		upd_in = torch.cat([agg, self_feat], dim=-1)  # (N, H + self_feat)
		upd = self.update_mlp(upd_in)  # (N, 2 + 1 + C + 1)

		dxdy = upd[:, 0:2]
		dtheta = upd[:, 2:3]
		dmol = upd[:, 3:3 + self.molecule_dim]
		divide_logit = upd[:, 3 + self.molecule_dim: 4 + self.molecule_dim]

		return dxdy, dtheta, dmol, divide_logit


def _quick_test():
	torch.manual_seed(0)
	N = 128
	C = 16
	x = torch.rand(N, 2)
	angle = torch.rand(N, 1) * (2 * math.pi) - math.pi
	mol = torch.randn(N, C)
	nca = ParticleNCA(molecule_dim=C, k=16, cutoff=0.2)
	gen = torch.zeros(N, 1)
	dxdy, dtheta, dmol, div_logit = nca(x, angle, mol, gen)
	assert dxdy.shape == (N, 2)
	assert dtheta.shape == (N, 1)
	assert dmol.shape == (N, C)
	assert div_logit.shape == (N, 1)
	print("ParticleNCA forward OK:", dxdy.mean().item(), dtheta.mean().item(), dmol.mean().item(), div_logit.mean().item())


if __name__ == "__main__":
	_quick_test()

