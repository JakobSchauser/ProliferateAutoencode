# Particle NCA — Design & Extension Notes

This document elaborates on the architecture and offers guidance for extending and training the model.

## Overview

- Entities: particles with features `(x, y)`, `angle`, and `molecules (C)`.
- Neighborhood: k-NN with cutoff. We mask neighbors beyond `cutoff` distance, so aggregation respects proximity.
- Messaging: all messages are relative — `(dx, dy, r)`, `d_angle`, and `(mol_j - mol_i)`. Optional Fourier features for distance; angles can be represented as sin/cos for continuity.
- Update: aggregate messages (sum or mean), concatenate with self features, predict deltas `(dx, dy, d_angle, d_molecules)`.

## Model Interface

 Entities: particles with features `(x, y)`, `angle`, `molecules (C)`, and `generation (scalar)`.

 Messaging: all messages are relative — `(dx, dy, r)`, `d_angle`, `(mol_j - mol_i)`, and `(gen_j - gen_i)`. Optional Fourier features for distance; angles can be represented as sin/cos for continuity.
- `molecule_dim: int` — number of hidden channels
 Update: aggregate messages (sum or mean), concatenate with self features, predict deltas `(dx, dy, d_angle, d_molecules)`, and a `divide_logit` decision.
- `cutoff: float = 0.25` — distance threshold
- `message_hidden: int = 64`, `update_hidden: int = 64`
- `aggregate: str = "sum"` — or `"mean"`
- `positional_encoding: bool = True` — add Fourier features for distances
- `angle_sin_cos: bool = True` — encode angles via sin/cos
(dxdy, dtheta, dmol, divide_logit) = nca(x, angle, molecules, generation)
Forward:
# x: (N,2), angle: (N,1), molecules: (N,C), generation: (N,1)
```
# dxdy: (N,2), dtheta: (N,1), dmol: (N,C), divide_logit: (N,1)
(dxdy, dtheta, dmol) = nca(x, angle, molecules)
# x: (N,2), angle: (N,1), molecules: (N,C)
```

## Implementation Details
  - Generation: `gen_j - gen_i`

 Update MLP: consumes aggregated message + self features (angle encoding + molecules + generation) to produce deltas and division decision.
- k-NN selection: `topk` over pairwise distances finds nearest neighbors. We create a boolean mask to flag neighbors within cutoff.
- Relative features per edge:
  - Geometry: `dx, dy, r`, optionally `PE(r)` from two-frequency Fourier features
  - Angle: either `sin(d_angle), cos(d_angle)` or raw `d_angle` (with optional Fourier augment)
  - Molecules: `mol_j - mol_i`
dxdy, dtheta, dmol, divide_logit = nca(x, angle, mol, gen)
- Aggregation: sum or mean over valid edges using the cutoff mask.
- Update MLP: consumes aggregated message + self features (angle encoding + molecules) to produce deltas.

divide = (torch.sigmoid(divide_logit) > 0.5)
gen = gen + divide.float()
```
Division policy:
- When `divide` is true, duplicate the particle (or spawn a new one) with:
  - Position/angle/molecules initialized from parent (optionally with small noise)
  - `generation_child = generation_parent + 1`
  - Parent may optionally remain or be split depending on the biological analogy.
- Simulation allows overshooting beyond the target cell count (4096) to let the loss guide behavior.
## Simulation Loop

Typical step:
```
- When training division, consider focal loss or class weighting for rare events.
- To stabilize, gate gradients through discrete division via straight-through estimators or train with soft decisions first.
mol = mol + dmol
```
Consider clamping or wrap-around for `angle`:
```
angle = (angle + math.pi) % (2*math.pi) - math.pi
```

## Training Sketch

Objective depends on your task (e.g., autoencoding target particle shapes, dynamics matching). A simple sketch:
```
optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)
for step in range(steps):
    dxdy, dtheta, dmol = nca(x, angle, mol)
    x, angle, mol = x + dxdy, angle + dtheta, mol + dmol
    loss = compute_loss(x, angle, mol, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
Tips:
- Normalize inputs (positions, angles) to consistent ranges.
- Use gradient clipping.
- Curriculum on `k` and `cutoff` can stabilize learning.

## Extensions

- Learned cutoff: predict per-node or per-edge gating weights instead of hard cutoff mask.
- Attention-style aggregation: replace sum/mean with softmax-weighted aggregation based on learned scores.
- Vector and orientation-aware messages: rotate `(dx, dy)` into the local frame using `angle_i`.
- Multi-step rollouts: unroll several steps and backprop through time.
- Molecule interactions: add reaction terms or self-update residuals.

## Performance Notes

- Complexity: pairwise distance is `O(N^2)`; suitable for moderate `N` (<= few thousands). For larger `N`, consider approximate k-NN (faiss) or spatial hashing.
- Batched worlds: extend inputs with batch dimension `(B, N, ...)` and adjust k-NN to operate per batch.

## Testing

- `src/NCAArchitecture.py` includes `_quick_test()` for shape sanity.
- Add property-based tests for invariants (e.g., zero message when mask is false).

## License & Attribution

- No license headers added; adapt as needed.
