import math
import random
from dataclasses import dataclass, asdict, fields
from typing import Callable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from NCAArchitecture2 import ParticleNCA_edge

from target_functions import looks_like_vitruvian, looks_like_vitruvian_gaussian, looks_like_vitruvian_gaussian_rotated, color_and_cover
from biological_coarse_graining.coarse_grain import sample_positions_from_image, image_paths


@dataclass
class GAConfig:
    population_size: int = 16
    elite_frac: float = 0.25
    mutation_std: float = 0.05
    crossover_frac: float = 0.5
    generations: int = 200
    rollout_steps: int = 200
    dt: float = 0.1
    target_count: int = 4096
    max_cells: int = 10000  # Allow overshoot beyond target
    device: str = "cpu"
    n_molecules: int = 16
    k: int = 16
    cutoff: float = 0.25
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
    cell_size: float = 0.1
    noise_eta: float = 1e-4 # magnitude of gaussian noise injected into updates
    # Note: global rotation angle is chosen per evaluation run

class ParticleWorld:
    def __init__(self, cfg: GAConfig, model: ParticleNCA_edge, global_angle: float = 0.0):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.x = None
        self.angle = None
        self.mol = None
        self.gen = None
        self.global_angle = float(global_angle)


    def add_globals(self):
        x,y = np.cos(self.global_angle), np.sin(self.global_angle)
        self.mol[:, 0] = x  # global angle cos
        self.mol[:, 1] = y  # global angle sin
        # self.mol[:, 0] = self.x[:, 0] #* 2.0 - 1.0  # normalized x pos
        # self.mol[:, 1] = self.x[:, 1]# * 2.0 - 1.0  # normalized y pos

    def add_first_time_globals(self):   
        xx, yy = self.x[:,0], self.x[:,1]
        self.mol[:, 2] = (xx - xx.min())/(xx.max() - xx.min())  # normalized x pos
        self.mol[:, 3] = (yy - yy.min())/(yy.max() - yy.min())  # normalized y pos
        self.add_globals()

    def reset(self, n0: int = 1):
        self.x = torch.zeros(n0, 2, device=self.device)
        self.angle = torch.zeros(n0, 1, device=self.device)
        self.mol = torch.zeros(n0, self.cfg.n_molecules, device=self.device)
        self.gen = torch.zeros(n0, 1, device=self.device)
        # Encode per-run global angle as a dedicated molecule channel (index 0)
        # self.add_globals()
        self.add_first_time_globals()


    def reset_to_positions(self, level : int):
        pos = sample_positions_from_image(level)
        positions = torch.tensor(pos, dtype=torch.float32, device=self.device)
        # positions = torch.zeros_like(pos)
        N = pos.shape[0]
        self.x = positions.clone()
        # self.x = positions.to(self.device)
        self.angle = torch.zeros(N, 1, device=self.device)
        self.mol = torch.zeros(N, self.cfg.n_molecules, device=self.device)
        self.gen = torch.zeros(N, 1, device=self.device)
        # Encode per-run global angle as a dedicated molecule channel (index 0)
        # self.add_globals()
        self.add_first_time_globals()

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
        self.gen[sel] += 1.0
        gen_child = self.gen[sel].clone()
        self.x = torch.cat([self.x, x_child], dim=0)
        self.angle = torch.cat([self.angle, angle_child], dim=0)
        self.mol = torch.cat([self.mol, mol_child], dim=0)
        self.gen = torch.cat([self.gen, gen_child], dim=0)
        # Ensure global angle channel remains constant for new cells
        self.add_globals()
        
    def step(self):
        dxdy, dtheta, dmol, div_logit = self.model(self.x, self.angle, self.mol, self.gen)
        eta = float(self.cfg.noise_eta)
        dxdy = dxdy + torch.randn_like(dxdy) * eta
        dtheta = dtheta + torch.randn_like(dtheta) * eta
        dmol = dmol + torch.randn_like(dmol) * eta
        self.x = self.x + dxdy * self.cfg.dt
        self.angle = self.angle + dtheta * self.cfg.dt
        self.mol = self.mol + dmol * self.cfg.dt
        # Keep global angle molecule channel immutable across steps
        divide = (torch.sigmoid(div_logit) > 0.5).squeeze(-1)
        # self._perform_divisions(divide)
        self.add_globals()



def clone_model(model: ParticleNCA_edge) -> ParticleNCA_edge:
    clone = ParticleNCA_edge(
        molecule_dim=model.molecule_dim,
        k=model.k,
        cutoff=model.cutoff,
    )
    clone.load_state_dict(model.state_dict())
    return clone


def mutate_model(model: ParticleNCA_edge, std: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * std
            p.add_(noise)


def crossover_models(parent_a: ParticleNCA_edge, parent_b: ParticleNCA_edge, frac: float) -> ParticleNCA_edge:
    child = clone_model(parent_a)
    with torch.no_grad():
        for (name_c, p_c), (_, p_a), (_, p_b) in zip(child.named_parameters(), parent_a.named_parameters(), parent_b.named_parameters()):
            mask = torch.rand_like(p_c) < frac
            p_c.copy_(torch.where(mask, p_a, p_b))
    return child


def _save_checkpoint(cfg: GAConfig, generation: int, population: List[ParticleNCA_edge], history: List[float]):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.name}_{generation}.pt")
    state = {
        "generation": generation,
        "history": history,
        "cfg": asdict(cfg),
        "population": [m.state_dict() for m in population],
        "model_meta": {
            "molecule_dim": cfg.n_molecules,
            "k": cfg.k,
            "cutoff": cfg.cutoff,
            "rollout_steps": cfg.rollout_steps,
        },
    }
    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved population to {ckpt_path}")


def _load_checkpoint(cfg: GAConfig) -> Tuple[int, List[ParticleNCA_edge], List[float]]:
    assert cfg.resume_path is not None
    state = torch.load(cfg.resume_path, map_location="cpu")
    start_gen = int(state.get("generation", 0))
    history = list(state.get("history", []))
    # If a serialized cfg exists, update the provided cfg in-place to match it
    saved_cfg_dict = state.get("cfg", None)
    if isinstance(saved_cfg_dict, dict):
        cfg_field_names = {f.name for f in fields(GAConfig)}
        for k, v in saved_cfg_dict.items():
            if k in cfg_field_names:
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass
    meta = state.get("model_meta", {})
    population_sd = state.get("population", [])
    population: List[ParticleNCA_edge] = []
    for sd in population_sd:
        m = ParticleNCA_edge(
            molecule_dim=int(meta.get("molecule_dim", cfg.n_molecules)),
            k=int(meta.get("k", cfg.k)),
            cutoff=float(meta.get("cutoff", cfg.cutoff)),
        )
        m.load_state_dict(sd)
        population.append(m)
    print(f"[Checkpoint] Loaded population from {cfg.resume_path} (gen {start_gen})")
    return start_gen, population, history

    
from target_functions import get_all_cells_to_origin_distance
# # Select fitness function: 
def fitness_fn(world, cfg):
    # Rotated target using the per-run global angle
    # return get_all_cells_to_origin_distance(world, cfg)
    return looks_like_vitruvian_gaussian_rotated(world, cfg, level=1, gauss_width=cfg.cell_size, threshold=0.85, angle_rad=world.global_angle)
    # return color_and_cover(world, cfg, level=0, gauss_width=cfg.cell_size, threshold=0.85, angle_rad=world.global_angle)
    # return looks_like_vitruvian_gaussian_rotated_colored(world, cfg, level=0, gauss_width=cfg.cell_size, threshold=0.85, angle_rad=world.global_angle)



from target_functions import color_fitness, color_fitness_rotated
# Select fitness function: 
# def fitness_fn(world, cfg):
#     return color_fitness_rotated(world, cfg, angle_rad=world.global_angle)


def evaluate_model(
    cfg: GAConfig,
    model: ParticleNCA_edge,
    N_times: int = 1,
    return_positions: bool = False,
    return_model_states: bool = False,
    record_interval: int = 1,
    return_fitness_per_frame: bool = False,
    global_angle: Optional[float] = None,
):
    """
    Evaluate a model for cfg.rollout_steps. Optionally return rollout positions.

    - When return_positions=False (default): returns float fitness (as before).
    - When return_positions=True: returns a tuple (fitness, positions, fitnesses_per_frame?) where
        positions is List[Tensor] sampled every `record_interval` steps. If
        return_fitness_per_frame=True, also returns a list of per-sample fitness values; otherwise []

    Note: N_times controls how many times the fitness is sampled within ONE rollout.
    We sample the fitness N_times uniformly across the rollout and average those samples.
    """
    # Choose per-run global angle if not provided
    world = ParticleWorld(cfg, model, global_angle=global_angle)
    # world.reset(n0=1)

    # positions_from
    world.reset_to_positions(level = 0)

    record_positions: List[torch.Tensor] = []
    record_states: List = []
    record_fitness: List[float] = []

    # Determine sampling steps for fitness: N_times uniformly spaced over rollout
    n_samples = max(1, int(N_times))
    if n_samples >= cfg.rollout_steps:
        sample_steps = set(range(1, cfg.rollout_steps + 1))
    else:
        # Use evenly spaced indices in 1..rollout_steps
        sample_steps = set(
            max(1, min(cfg.rollout_steps, int(round(s))))
            for s in torch.linspace(cfg.rollout_steps//2, cfg.rollout_steps, steps=n_samples).tolist()
        )

    total_fitness = 0.0
    for t in range(cfg.rollout_steps):
        world.step()
        if return_positions and ((t + 1) % record_interval == 0):
            if world.x is not None:
                record_positions.append(world.x.detach().clone())
                states = np.hstack((world.angle.detach().clone(), world.mol.detach().clone()))
                record_states.append(states)
                if return_fitness_per_frame:
                    record_fitness.append(fitness_fn(world, cfg))
        # sample fitness at selected steps
        if (t + 1) in sample_steps:
            total_fitness += float(fitness_fn(world, cfg))

    avg_fitness = float(total_fitness / max(1, n_samples))

    ret = [avg_fitness]
    if return_model_states:
        ret.append(record_states)
    if return_positions:
        ret.append(record_positions)
    if return_fitness_per_frame:
        ret.append(record_fitness)

    return tuple(ret)

def genetic_train(cfg: GAConfig, use_threads: bool = True, max_workers: Optional[int] = None, random_start_angle: bool = False) -> Tuple[ParticleNCA_edge, List[float]]:
    # Initialize or resume population
    if cfg.resume_path:
        start_gen, population, history = _load_checkpoint(cfg)
    else:
        start_gen = 0
        population: List[ParticleNCA_edge] = []
        for _ in range(cfg.population_size):
            m = ParticleNCA_edge(
                molecule_dim=cfg.n_molecules,
                k=cfg.k,
                cutoff=cfg.cutoff,
            )
            population.append(m)
        history: List[float] = []

    elite_k = max(1, int(cfg.elite_frac * cfg.population_size))

    for g in range(start_gen, cfg.generations):
        print(f"[GA] Generation {g+1}/{cfg.generations} — evaluating {len(population)} individuals...")
        scores: List[float] = [float('-inf')] * len(population)
        if use_threads:
            mw = min(16, len(population)) if max_workers is None else min(max_workers, len(population))
            print(f"[GA] Using threaded evaluation with {mw} worker(s).")
            completed = 0
            ex = ThreadPoolExecutor(max_workers=mw)
            try:
                future_to_idx = {ex.submit(evaluate_model, cfg, m, cfg.N_times): i for i, m in enumerate(population)}
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    res = fut.result()
                    scores[i] = float(res if not isinstance(res, tuple) else res[0])

                    completed += 1
                    if completed % max(1, len(population)//4) == 0 or completed == len(population):
                        print(f"[GA] Evaluation progress: {completed}/{len(population)} done")
            except KeyboardInterrupt:
                print("[GA] KeyboardInterrupt received. Cancelling pending evaluations and shutting down threads...")
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                ex.shutdown(wait=True)
        else:
            print("[GA] Using sequential evaluation.")
            ang = random.uniform(0, 2*math.pi) if random_start_angle else 0.0 
            for i, m in enumerate(population):
                res = evaluate_model(cfg, m, cfg.N_times, global_angle=ang)
                scores[i] = float(res if not isinstance(res, tuple) else res[0])
                if (i+1) % max(1, len(population)//4) == 0 or (i+1) == len(population):
                    print(f"[GA] Evaluation progress: {i+1}/{len(population)} done")
        best_score_gen = max(scores)
        history.append(best_score_gen)
        print(f"[GA] Generation {g+1} evaluation complete. Best score={best_score_gen:.4f}")

        # Select elites
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elites = [clone_model(m) for _, m in ranked[:elite_k]]

        # Reproduce
        # Order: elites, 2 random, mutated elites (2 per elite)
        new_pop: List[ParticleNCA_edge] = elites.copy()

        # Add up to 1 fresh random individuals each generation
        for _ in range(1):
            if len(new_pop) >= cfg.population_size:
                break
            m_rand = ParticleNCA_edge(
                molecule_dim=cfg.n_molecules,
                k=cfg.k,
                cutoff=cfg.cutoff,
            )
            new_pop.append(m_rand)

        # Each elite gets one direct mutated descendant
        for e in elites:
            if len(new_pop) >= cfg.population_size:
                break
            child = clone_model(e)
            mutate_model(child, cfg.mutation_std)
            new_pop.append(child)

        # # Fill the remaining slots with crossover children from elites, then mutate
        while len(new_pop) < cfg.population_size:
            a = random.choice(elites)
            b = random.choice(elites)
            child = crossover_models(a, b, cfg.crossover_frac)
            mutate_model(child, cfg.mutation_std)
            new_pop.append(child)

        # # fill the remaining slots with more random children from elites
        # while len(new_pop) < cfg.population_size:
        #     e = random.choice(elites)
        #     child = clone_model(e)
        #     mutate_model(child, cfg.mutation_std*5.)
        #     new_pop.append(child)


        population = new_pop

        best = ranked[0][0]
        worst = ranked[-1][0]
        median_elite = ranked[elite_k // 2][0]
        print(f"Gen {g+1}/{cfg.generations} | best_fitness={best:.2f} | worst_fitness={worst:.2f} | median_elite_fitness={median_elite:.2f}")

        # Save checkpoint every 10 generations
        if (g + 1) % 5 == 0:
            tp = int((g + 1) / 50)*50
            _save_checkpoint(cfg, tp, population, history)

    # Return best individual
    final_scores = []
    for m in population:
        res = evaluate_model(cfg, m)
        final_scores.append(float(res if not isinstance(res, tuple) else res[0]))
    best_idx = int(torch.tensor(final_scores).argmax().item())
    return population[best_idx], history
