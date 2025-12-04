import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from NCAArchitecture import ParticleNCA


@dataclass
class GAConfig:
    population_size: int = 16
    elite_frac: float = 0.25
    mutation_std: float = 0.05
    crossover_frac: float = 0.5
    generations: int = 200
    rollout_steps: int = 1000
    dt: float = 0.1
    target_count: int = 4096
    max_cells: int = 10000  # Allow overshoot beyond target
    device: str = "cpu"
    n_molecules: int = 16
    k: int = 16
    cutoff: float = 0.25
    angle_sin_cos: bool = True
    positional_encoding: bool = True
    aggregate: str = "sum"
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_path: Optional[str] = None
    # Fitness shaping
    circle_weight: float = 0.0  # weight for circle uniformity term
    name : str = "ga_gen"
    # Image/emoji target for shape-based fitness
    image: Optional[str] = None
    emoji: Optional[str] = "🐣"
    N_times : int = 1  # Number of times to evaluate each model and average fitness


class ParticleWorld:
    def __init__(self, cfg: GAConfig, model: ParticleNCA):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.x = None
        self.angle = None
        self.mol = None
        self.gen = None

    def reset(self, n0: int = 1):
        self.x = torch.zeros(n0, 2, device=self.device)
        self.angle = torch.zeros(n0, 1, device=self.device)
        self.mol = torch.zeros(n0, self.cfg.n_molecules, device=self.device)
        self.gen = torch.zeros(n0, 1, device=self.device)

    @torch.no_grad()
    def _perform_divisions(self, divide_mask: torch.Tensor):
        if divide_mask.sum() == 0:
            return
        N = self.x.shape[0]
        to_divide_idx = torch.nonzero(divide_mask, as_tuple=False).squeeze(-1)
        num_new = int(to_divide_idx.numel())
        if num_new <= 0:
            return
        # Cap total cells
        new_total = min(N + num_new, self.cfg.max_cells)
        num_new = max(0, new_total - N)
        if num_new == 0:
            return
        sel = to_divide_idx[:num_new]
        jitter_pos = 0.01
        jitter_ang = 0.01
        x_child = self.x[sel] + (torch.randn_like(self.x[sel]) * jitter_pos)
        angle_child = self.angle[sel] + (torch.randn_like(self.angle[sel]) * jitter_ang)
        mol_child = self.mol[sel]
        gen_child = self.gen[sel] + 1.0
        self.x = torch.cat([self.x, x_child], dim=0)
        self.angle = torch.cat([self.angle, angle_child], dim=0)
        self.mol = torch.cat([self.mol, mol_child], dim=0)
        self.gen = torch.cat([self.gen, gen_child], dim=0)

    def step(self):
        dxdy, dtheta, dmol, div_logit = self.model(self.x, self.angle, self.mol, self.gen)
        self.x = self.x + dxdy * self.cfg.dt
        self.angle = self.angle + dtheta * self.cfg.dt
        self.mol = self.mol + dmol * self.cfg.dt
        divide = (torch.sigmoid(div_logit) > 0.5).squeeze(-1)
        self._perform_divisions(divide)



def clone_model(model: ParticleNCA) -> ParticleNCA:
    clone = ParticleNCA(
        molecule_dim=model.molecule_dim,
        k=model.k,
        cutoff=model.cutoff,
        aggregate=model.aggregate,
        positional_encoding=model.positional_encoding,
        angle_sin_cos=model.angle_sin_cos,
    )
    clone.load_state_dict(model.state_dict())
    return clone


def mutate_model(model: ParticleNCA, std: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * std
            p.add_(noise)


def crossover_models(parent_a: ParticleNCA, parent_b: ParticleNCA, frac: float) -> ParticleNCA:
    child = clone_model(parent_a)
    with torch.no_grad():
        for (name_c, p_c), (_, p_a), (_, p_b) in zip(child.named_parameters(), parent_a.named_parameters(), parent_b.named_parameters()):
            mask = torch.rand_like(p_c) < frac
            p_c.copy_(torch.where(mask, p_a, p_b))
    return child


def _save_checkpoint(cfg: GAConfig, generation: int, population: List[ParticleNCA], history: List[float]):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.name}_{generation}.pt")
    state = {
        "generation": generation,
        "history": history,
        "population": [m.state_dict() for m in population],
        "model_meta": {
            "molecule_dim": cfg.n_molecules,
            "k": cfg.k,
            "cutoff": cfg.cutoff,
            "aggregate": cfg.aggregate,
            "positional_encoding": cfg.positional_encoding,
            "angle_sin_cos": cfg.angle_sin_cos,
        },
    }
    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved population to {ckpt_path}")


def _load_checkpoint(cfg: GAConfig) -> Tuple[int, List[ParticleNCA], List[float]]:
    assert cfg.resume_path is not None
    state = torch.load(cfg.resume_path, map_location="cpu")
    start_gen = int(state.get("generation", 0))
    history = list(state.get("history", []))
    meta = state.get("model_meta", {})
    population_sd = state.get("population", [])
    population: List[ParticleNCA] = []
    for sd in population_sd:
        m = ParticleNCA(
            molecule_dim=int(meta.get("molecule_dim", cfg.n_molecules)),
            k=int(meta.get("k", cfg.k)),
            cutoff=float(meta.get("cutoff", cfg.cutoff)),
            angle_sin_cos=bool(meta.get("angle_sin_cos", cfg.angle_sin_cos)),
            positional_encoding=bool(meta.get("positional_encoding", cfg.positional_encoding)),
            aggregate=str(meta.get("aggregate", cfg.aggregate)),
        )
        m.load_state_dict(sd)
        population.append(m)
    print(f"[Checkpoint] Loaded population from {cfg.resume_path} (gen {start_gen})")
    return start_gen, population, history


from target_functions import correct_cell_count_fitness, looks_like_image_fitness

# Select fitness function: use emoji/image-based shape matching if configured
def fitness_fn(world, cfg):
    if cfg.image is not None or cfg.emoji is not None:
        return looks_like_image_fitness(world, cfg, image=cfg.image, emoji=cfg.emoji)
    return correct_cell_count_fitness(world, cfg)

def evaluate_model(cfg: GAConfig, model: ParticleNCA, N_times : int= 1) -> float:
  world = ParticleWorld(cfg, model)
  world.reset(n0=1)

  fitness = 0.0
  for iii in range(cfg.rollout_steps):
    world.step()
    if (iii+1) % (cfg.rollout_steps // N_times) == 0:
        fitness += fitness_fn(world, cfg)

    fitness /= N_times
  # Fitness: negative squared distance from target count (higher is better)
  return float(fitness)

def genetic_train(cfg: GAConfig) -> Tuple[ParticleNCA, List[float]]:
    # Initialize or resume population
    if cfg.resume_path:
        start_gen, population, history = _load_checkpoint(cfg)
    else:
        start_gen = 0
        population: List[ParticleNCA] = []
        for _ in range(cfg.population_size):
            m = ParticleNCA(
                molecule_dim=cfg.n_molecules,
                k=cfg.k,
                cutoff=cfg.cutoff,
                angle_sin_cos=cfg.angle_sin_cos,
                positional_encoding=cfg.positional_encoding,
                aggregate=cfg.aggregate,
            )
            population.append(m)
        history: List[float] = []

    elite_k = max(1, int(cfg.elite_frac * cfg.population_size))

    for g in range(start_gen, cfg.generations):
        # Evaluate population
        scores = [evaluate_model(cfg, m, N_times=cfg.N_times) for m in population]
        history.append(max(scores))

        # Select elites
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elites = [clone_model(m) for _, m in ranked[:elite_k]]

        # Reproduce
        new_pop: List[ParticleNCA] = elites.copy()
        while len(new_pop) < cfg.population_size:
            # Pick two parents from elites
            a = random.choice(elites)
            b = random.choice(elites)
            child = crossover_models(a, b, cfg.crossover_frac)
            mutate_model(child, cfg.mutation_std)
            new_pop.append(child)

        population = new_pop

        best = ranked[0][0]
        worst = ranked[-1][0]
        median_elite = ranked[elite_k // 2][0]
        print(f"Gen {g+1}/{cfg.generations} | best_fitness={best:.2f} | worst_fitness={worst:.2f} | median_elite_fitness={median_elite:.2f}")

        # Save checkpoint every 5 generations
        if (g + 1) % 5 == 0:
            _save_checkpoint(cfg, g + 1, population, history)

    # Return best individual
    final_scores = [evaluate_model(cfg, m) for m in population]
    best_idx = int(torch.tensor(final_scores).argmax().item())
    return population[best_idx], history
