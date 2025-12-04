import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import knn


class ParticleNCAMP(MessagePassing):
    """
    PyTorch Geometric implementation of particle-based NCA.
    - Node features: positions x (N,2), angle (N,1), molecules (N,C), generation (N,1)
    - Edge features: relative dx, dy, r, d_angle (sin/cos optional), d_molecules
      (no generation as edge feature per request)
    - Aggregation: sum or mean
    - Outputs per node: dxdy, dtheta, dmol, divide_logit
    """

    def __init__(
        self,
        molecule_dim: int,
        k: int = 16,
        cutoff: float = 0.25,
        message_hidden: int = 64,
        update_hidden: int = 64,
        aggregate: str = "sum",
        positional_encoding: bool = True,
        angle_sin_cos: bool = True,
    ):
        super().__init__(aggr=aggregate)  # 'sum' or 'mean'
        self.molecule_dim = molecule_dim
        self.k = k
        self.cutoff = cutoff
        self.positional_encoding = positional_encoding
        self.angle_sin_cos = angle_sin_cos

        rel_geom_dim = 3  # dx, dy, r
        if self.positional_encoding:
            rel_geom_dim += 4  # Fourier features for r
        angle_dim = 2 if self.angle_sin_cos else 1
        rel_in_dim = rel_geom_dim + angle_dim + molecule_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(rel_in_dim, message_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(message_hidden, message_hidden),
            nn.ReLU(inplace=True),
        )

        self_feat_dim = (2 if self.angle_sin_cos else 1) + molecule_dim + 1  # + generation
        upd_in = message_hidden + self_feat_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(upd_in, update_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(update_hidden, update_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(update_hidden, 2 + 1 + molecule_dim + 1),
        )

    @staticmethod
    def _fourier_features(t: torch.Tensor, n_freqs: int = 2) -> torch.Tensor:
        outs = []
        for k in range(n_freqs):
            f = 2.0 ** k
            outs.append(torch.sin(f * t))
            outs.append(torch.cos(f * t))
        return torch.cat(outs, dim=-1)

    def forward(
        self,
        x: torch.Tensor,         # (N,2)
        angle: torch.Tensor,     # (N,1)
        mol: torch.Tensor,       # (N,C)
        gen: torch.Tensor,       # (N,1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build kNN edge_index globally
        N = x.shape[0]
        device = x.device
        # torch_geometric.utils.knn returns pairs (row, col) for neighbors of src in y
        # We'll use x as both src and y, excluding self by later masking
        row, col = knn(x, x, k=min(self.k, N))  # shape (E,)
        # Exclude self-edges
        self_mask = (row != col)
        row = row[self_mask]
        col = col[self_mask]

        # Cutoff mask
        d = torch.sqrt(torch.clamp(((x[row] - x[col]) ** 2).sum(-1), min=1e-12))
        cutoff_mask = (d <= self.cutoff)
        row = row[cutoff_mask]
        col = col[cutoff_mask]

        # Compute messages
        msg = self.propagate(
            edge_index=torch.stack([row, col], dim=0),
            x=x, angle=angle, mol=mol, gen=gen
        )  # (N, H)

        # Update per node
        if self.angle_sin_cos:
            self_ang = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        else:
            self_ang = angle
        self_feat = torch.cat([self_ang, mol, gen], dim=-1)
        upd_in = torch.cat([msg, self_feat], dim=-1)
        upd = self.update_mlp(upd_in)
        dxdy = upd[:, 0:2]
        dtheta = upd[:, 2:3]
        dmol = upd[:, 3:3 + self.molecule_dim]
        divide_logit = upd[:, 3 + self.molecule_dim: 4 + self.molecule_dim]
        return dxdy, dtheta, dmol, divide_logit

    def message(self, x_i, x_j, angle_i, angle_j, mol_i, mol_j):
        dxdy = x_j - x_i
        r = torch.sqrt(torch.clamp((dxdy ** 2).sum(-1, keepdim=True), min=1e-12))
        d_angle = angle_j - angle_i
        if self.angle_sin_cos:
            ang_feat = torch.cat([torch.sin(d_angle), torch.cos(d_angle)], dim=-1)
        else:
            ang_feat = d_angle
        d_mol = mol_j - mol_i
        if self.positional_encoding:
            r_pe = self._fourier_features(r)
            rel_geom = torch.cat([dxdy, r, r_pe], dim=-1)
        else:
            rel_geom = torch.cat([dxdy, r], dim=-1)
        rel_in = torch.cat([rel_geom, ang_feat, d_mol], dim=-1)
        return self.message_mlp(rel_in)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Respect base class aggregation (sum or mean)
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def __repr__(self):
        return f"ParticleNCAMP(C={self.molecule_dim}, k={self.k}, cutoff={self.cutoff})"
