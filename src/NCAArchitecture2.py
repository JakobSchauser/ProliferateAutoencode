import math
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class ParticleNCA_edge(nn.Module):
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
		heads: int = 2,
	):
		super().__init__()
		self.molecule_dim = molecule_dim
		self.k = k
		self.cutoff = cutoff
		self.heads = heads

		# Message MLP takes relative inputs
		# Inputs per edge:
		# - dx, dy, r
		# - sin(d_angle), cos(d_angle) or raw d_angle
		# - (mol_j - mol_i) molecules difference and mol_i
		rel_geom_dim = 3  # dx, dy, r

		angle_dim = 2
		rel_input_dim = rel_geom_dim + angle_dim + molecule_dim

		self.message_mlp = nn.Sequential(
			nn.Linear(rel_input_dim, message_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(message_hidden, message_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(message_hidden, message_hidden),
			nn.ReLU(inplace=True),
		)

		# Node feature dimensionality (used as node input to attention conv)
		# Components: sin(angle), cos(angle), molecules, generation, degree (n-connections)
		self.self_feat_dim = 2 + molecule_dim + 1 + 1
		# Edge attributes passed into attention conv: encoded edge + self (dst) features
		edge_attr_dim = message_hidden + self.self_feat_dim
		# Attention-based message passing that consumes edge_attr
		self.attn_conv = TransformerConv(
			in_channels=self.self_feat_dim,
			out_channels=update_hidden,
			heads=self.heads,
			concat=False,
			edge_dim=edge_attr_dim,
			dropout=0.0,
		)
		# Node head maps attended features to deltas
		self.node_head = nn.Sequential(
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

	def _hard_cutoff_neighbors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Symmetrized directed edges using distance cutoff."""
		with torch.no_grad():
			dist = self._pairwise_dist(x)
			N = dist.shape[0]
			dist = dist + torch.eye(N, device=x.device) * 1e6
			mask = dist <= self.cutoff
			# take only upper triangle to form undirected pairs, then duplicate to directed
			tri_mask = torch.triu(mask, diagonal=1)
			dst_u, src_u = torch.where(tri_mask)
			# duplicate to both directions
			src = torch.cat([src_u, dst_u], dim=0)
			dst = torch.cat([dst_u, src_u], dim=0)
		return src, dst
	
	def forward(
		self,
		x: torch.Tensor,           # (N, 2)
		angle: torch.Tensor,       # (N, 1)
		molecules: torch.Tensor,   # (N, molecule_dim)
		generation: torch.Tensor,  # (N, 1) integer-like counter
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		N = x.shape[0]

		src, dst = self._hard_cutoff_neighbors(x)  # (E,), (E,) where E = number of edges
		edge_index = torch.stack([src, dst], dim=0)  # (2, E)

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
		ang_feat = torch.cat([torch.sin(d_angle), torch.cos(d_angle)], dim=-1)  # (E,2)
		d_mol = mol_j - mol_i           # (E,C)

		rel_geom = torch.cat([dxdy, r], dim=-1)  # (E,3)

		# Per-edge input to message MLP
		rel_in = torch.cat([rel_geom, ang_feat, d_mol], dim=-1)  # (E + molecule_dim + angle_dim + rel_geom_dim)

		msg = self.message_mlp(rel_in)  # (E,H)

		# Self features for update head
		self_ang = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (N,2)
		# Degree encodes the number of neighbors (n-connections) for each node
		degree = torch.bincount(dst, minlength=N).float().unsqueeze(-1)
		self_feat = torch.cat([self_ang, molecules, generation, degree], dim=-1)  # (N, 2 + C + 1 + 1)

		# Edge attributes include encoded edge plus destination self features
		edge_attr = torch.cat([msg, self_feat[dst]], dim=-1)

		# Attention-based aggregation of edge-informed messages
		node_attn = self.attn_conv(self_feat, edge_index, edge_attr)  # (N, update_hidden)
		upd = self.node_head(node_attn)  # (N, 2 + 1 + C + 1)

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
	nca = ParticleNCA_edge(molecule_dim=C, k=16, cutoff=0.2)
	gen = torch.zeros(N, 1)
	dxdy, dtheta, dmol, div_logit = nca(x, angle, mol, gen)
	assert dxdy.shape == (N, 2)
	assert dtheta.shape == (N, 1)
	assert dmol.shape == (N, C)
	assert div_logit.shape == (N, 1)
	print("ParticleNCA forward OK:", dxdy.mean().item(), dtheta.mean().item(), dmol.mean().item(), div_logit.mean().item())


if __name__ == "__main__":
	_quick_test()

