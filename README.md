# EggAutoencoder NCA — Quick Start

This repo provides a particle-based Neural Cellular Automata (NCA) built with PyTorch. Each particle has position `(x,y)`, orientation `angle`, hidden channels `molecules`, and a `generation` counter. Neighborhoods are k-NN with a distance cutoff; messages are computed from relative features; the model outputs deltas `(dx, dy, dangle, d_molecules)` and a `divide_logit` decision.

## Install

- Requires Python 3.13 (or compatible) and PyTorch.
- If PyTorch isn't installed, follow: https://pytorch.org/get-started/locally/

## Try It

Run the built-in quick test:

```powershell
C:/Python313/python.exe src/NCAArchitecture.py
```

Or use it in your code:

```python
import math, torch
from src.NCAArchitecture import ParticleNCA

N, C = 128, 16
x = torch.rand(N, 2)
angle = torch.rand(N, 1) * (2*math.pi) - math.pi
mol = torch.randn(N, C)
gen = torch.zeros(N, 1)

nca = ParticleNCA(molecule_dim=C, k=16, cutoff=0.25)

dxdy, dtheta, dmol, divide_logit = nca(x, angle, mol, gen)
# Apply update
x = x + dxdy
angle = angle + dtheta
mol = mol + dmol
# Division decision (example):
divide = (torch.sigmoid(divide_logit) > 0.5).squeeze(-1)
```

## Key Options

- `k`: neighbors per particle (default 16)
- `cutoff`: max distance for valid neighbors
- `aggregate`: `"sum"` or `"mean"`
- `angle_sin_cos`: use sin/cos encodings for angles
- `positional_encoding`: add simple Fourier features to distances
- `generation`: passed into forward; increment externally upon division

## Files

- `src/NCAArchitecture.py`: NCA model with kNN + cutoff and relative messaging; outputs `divide_logit`
- `src/simulation.py`: (your code) integrate update steps into a loop
- `docs/README-dev.md`: deeper design notes and extension tips
- `src/genetic/`: non-differentiable training via a simple genetic algorithm
	- `simulation_genetic.py`: GA environment, evaluation, selection, crossover, mutation
	- `run_genetic.py`: minimal runner

## Train to Divide

Run the training script to proliferate from 1 to 4096 cells over 2000 epochs. The loss encourages reaching exactly 4096 cells but allows overshooting. Configurable via `SimConfig` in `simulation.py`.

```powershell
C:/Python313/python.exe src/run.py
```

The training loop is modular: swap the loss via `simulation.py`'s `loss_fn` argument.

## Genetic Training (non-differentiable)

Use the GA runner when you prefer a non-differentiable objective (e.g., exact count target):

```powershell
C:/Python313/python.exe src/genetic/run_genetic.py
```

Adjust `GAConfig` for population size, mutation, crossover, generations, and target count.
