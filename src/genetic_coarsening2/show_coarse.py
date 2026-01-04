import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from simulation_coarse import GAConfig, _load_checkpoint, evaluate_model
from NCAArchitecture import ParticleNCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from target_functions import sample_positions_from_image

def load_best_from_checkpoint(resume_path: str, rollout_steps: Optional[int] = 1):
    # Initialize with only resume_path; _load_checkpoint will hydrate cfg from checkpoint
    cfg = GAConfig(resume_path=resume_path)
    start_gen, population, history = _load_checkpoint(cfg)
    # Allow caller to override rollout length for evaluation/visualization
    if rollout_steps is not None:
        cfg.rollout_steps = rollout_steps
    print(f"Loaded checkpoint gen={start_gen}, population size={len(population)}, history len={len(history)}")
        # Evaluate all models in parallel and pick the one with highest fitness
    if not population:
        raise ValueError("Checkpoint contained an empty population")
    scores = [float('-inf')] * len(population)
    max_workers = min(16, len(population))
    n_times = max(1, getattr(cfg, 'rollout_steps', 1))
    # print("N_times for evaluation:", n_times)
    # with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #     future_to_idx = {ex.submit(evaluate_model, cfg, m, n_times): i for i, m in enumerate(population)}
    #     for fut in as_completed(future_to_idx):
    #         i = future_to_idx[fut]
    #         try:
    #             res = fut.result()
    #             scores[i] = float(res if not isinstance(res, tuple) else res[0])
    #         except Exception as e:
    #             scores[i] = float('-inf')
    #             print(f"[Warn] Evaluation failed for model {i}: {e}")
    # best_idx = max(range(len(scores)), key=lambda i: scores[i])
    # best_score = scores[best_idx]
    # print(f"Selected best model index={best_idx} with fitness={best_score:.4f}")
    best_idx = 0
    return population[best_idx], cfg


# Removed rollout_and_show; use evaluate_model return for visualization


_DEFAULT_N_MOLECULES = 16

def rollout_and_show_video(
    model: ParticleNCA,
    out_path: str = "rollout.mp4",
    fps: int = 30,
    s: int = 6,
    level: int = 1,
    state_index: int = 1,
    state_vmin: float = 0.0,
    state_vmax: float = 1.0,
    cmap: str = "viridis",
    show_colorbar: bool = True,
    cfg: Optional[GAConfig] = None,
):
    """
    Roll out the world and render an animation. Saves to MP4 if out_path ends with .mp4,
    or GIF if it ends with .gif. If out_path is None, defaults to 'rollout.mp4'.
    """
    # Use provided cfg (ideally restored from checkpoint); fallback to defaults
    if cfg is None:
        cfg = GAConfig()

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=s, alpha=0.8, cmap=cmap, zorder=2)
    norm = Normalize(vmin=state_vmin, vmax=state_vmax)
    scat.set_norm(norm)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("NCA Rollout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Optional colorbar showing state value mapping
    cbar = None
    if show_colorbar:
        cbar = plt.colorbar(scat, ax=ax)
        cbar.set_label(f"state[{state_index}]", rotation=90)

    # Prepare target overlay points once
    target_pts = sample_positions_from_image(level)
    target_artist = None

    # Use evaluate_model to get positions and per-frame fitness for visualization
    # Ensure rollout length is set
    if not isinstance(cfg.rollout_steps, int) or cfg.rollout_steps <= 0:
        cfg.rollout_steps = 300
    res = evaluate_model(
        cfg,
        model,
        N_times=1,
        return_positions=True,
        return_model_states=True,
        record_interval=1,
        return_fitness_per_frame=True,
    )
    if isinstance(res, tuple) and len(res) >= 3:
        # (avg_fitness, states_t, positions_t[, fitnesses])
        states_t = res[1]
        positions_t = res[2]
        fitnesses = res[3] if len(res) >= 4 else []
    else:
        states_t = []
        positions_t = []
        fitnesses = []
    positions = [p.detach().cpu().numpy() for p in positions_t] if isinstance(positions_t, (list, tuple)) else []
    # Color by selected molecule state channel per particle
    colors = []
    for st in states_t if isinstance(states_t, (list, tuple)) else []:
        if st is None:
            colors.append(np.array([], dtype=np.float32))
            continue
        ang_t, mol_t = st
        if mol_t is None or mol_t.numel() == 0:
            colors.append(np.array([], dtype=np.float32))
            continue
        if 0 <= state_index < mol_t.shape[1]:
            c = mol_t[:, state_index].detach().cpu().numpy().astype(np.float32)
        else:
            c = np.ones((mol_t.shape[0],), dtype=np.float32)
        # Clamp to [state_vmin, state_vmax] for colormap normalization
        c = np.clip(c, state_vmin, state_vmax)
        colors.append(c)

    # Compute global axis limits to keep all cells in frame
    valid = [p for p in positions if p.shape[0] > 0]
    if target_pts is not None and target_pts.shape[0] > 0:
        # Use normalized axes to correctly overlay target
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        target_artist = ax.scatter(target_pts[:, 0], target_pts[:, 1], s=4, alpha=0.25, color='black', zorder=1)
    else:
        if len(valid) > 0:
            all_pos = np.concatenate(valid, axis=0)
            xmin, ymin = all_pos.min(axis=0)
            xmax, ymax = all_pos.max(axis=0)
            pad_x = max(0.05, 0.1 * (xmax - xmin + 1e-6))
            pad_y = max(0.05, 0.1 * (ymax - ymin + 1e-6))
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
        else:
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        return (scat,) if target_artist is None else (target_artist, scat)

    def update(frame):
        pos = positions[frame]
        col = colors[frame] if frame < len(colors) else np.array([], dtype=np.float32)
        scat.set_offsets(pos)
        scat.set_array(col)
        f_list = fitnesses if isinstance(fitnesses, (list, tuple)) else []
        fit = f_list[frame] if frame < len(f_list) else None
        title = f"t={frame+1} (n={pos.shape[0]})"
        if fit is not None:
            title += f" | fitness={fit:.3f}"
        ax.set_title(title)
        return (scat,) if target_artist is None else (target_artist, scat)

    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    anim = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=1000//fps)

    out_path = out_path or os.path.join(os.path.dirname(__file__), "rollout.mp4")

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='ProliferateAutoencode'))
            anim.save(out_path, writer=writer)
        except Exception as e:
            print(f"FFMpeg not available ({e}); falling back to GIF...")
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            out_path = gif_path
    elif ext == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
    else:
        # default to mp4
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='ProliferateAutoencode'))
        anim.save(out_path, writer=writer)

    plt.close(fig)
    print(f"Saved rollout video to {out_path}")



def rollout_and_show_video_gaussian(
    model: ParticleNCA,
    out_path: str = "rollout_gaussian.mp4",
    fps: int = 30,
    level: int = 1,
    grid_res: int = 256,
    mode: str = "max",
    state_index: int = -1,
    cmap: str = "viridis",
    show_colorbar: bool = True,
    cfg: Optional[GAConfig] = None,
    draw_cutoff: bool = False,
    angle: Optional[float] = None,
):
    """
    Render an animation of the Gaussian field induced by particle positions.
    - Field is either the max or sum of per-particle Gaussians with width `gauss_width`.
    - Overlays target points normalized to [-1,1] with inverted Y to match plot coords.
    """
    # Use provided cfg; fallback to default
    if cfg is None:
        cfg = GAConfig()

    # Load target points and normalize to [-1, 1] for overlay
    target_pts = sample_positions_from_image(level)
    # If a global angle is provided, rotate target overlay accordingly
    if angle is not None and isinstance(target_pts, np.ndarray) and target_pts.shape[1] == 2:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        target_pts = (target_pts @ R.T).astype(np.float32)

    # Rollout to collect positions and model states per frame
    res = evaluate_model(
        cfg,
        model,
        N_times=1,
        return_positions=True,
        return_model_states=True,
        record_interval=1,
        return_fitness_per_frame=True,
        global_angle=angle,
    )

    gauss_width = cfg.cell_size
    states_t = res[1]
    positions_t = res[2]
    fitness_history = res[3]

    positions = [p.detach().cpu().numpy() for p in positions_t] if isinstance(positions_t, (list, tuple)) else []
        # Extract per-particle colors from selected molecule state channel
    # states_t entries are tuples: (angle_t, mol_t)

    # Determine axis limits
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    if target_pts is not None and target_pts.shape[0] > 0:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
    else:
        valid = [p for p in positions if p.shape[0] > 0]
        if len(valid) > 0:
            all_pos = np.concatenate(valid, axis=0)
            xmin, ymin = all_pos.min(axis=0)
            xmax, ymax = all_pos.max(axis=0)
            pad_x = max(0.05, 0.01 * (xmax - xmin + 1e-6))
            pad_y = max(0.05, 0.01 * (ymax - ymin + 1e-6))
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
        else:
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)

    # Build grid over current axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xs = np.linspace(xlim[0], xlim[1], grid_res, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_res, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (H*W, 2)

    # Initialize image artist for Gaussian field
    im = ax.imshow(
        np.zeros((grid_res, grid_res), dtype=np.float32),
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin='lower',
        cmap='inferno',
        vmin=0.0,
        vmax=1.0,
        zorder=1,
        interpolation='nearest',
    )
    # Overlay scatter of particles colored by selected state channel
    scat = ax.scatter([], [], s=6, alpha=0.8, cmap=cmap, zorder=3)
    state_vmin = np.min([s[:,state_index].min() for s in states_t])
    state_vmax = np.max([s[:,state_index].max() for s in states_t])
    print(f"State[{state_index}] value range over rollout: [{state_vmin:.4f}, {state_vmax:.4f}]")
    norm = Normalize(vmin=state_vmin, vmax=state_vmax)
    scat.set_norm(norm)
    # Optional colorbar for state values (place before animation so it's included)
    cbar = None
    if show_colorbar:
        cbar = plt.colorbar(scat, ax=ax)
        cbar.set_label(f"state[{state_index}]", rotation=90)
    plt.tight_layout()

    # Prepare cutoff circles overlay
    cutoff_artist = None

    # rotate target points for overlay
    from target_functions import rotate_positions
    target_pts = rotate_positions(target_pts, angle)

    # Overlay targets, if available
    if target_pts is not None and target_pts.shape[0] > 0:
        target_artist = ax.scatter(target_pts[:, 0], target_pts[:, 1], s=6, alpha=0.3, color='white', zorder=2)

    ax.set_title("Gaussian Field Rollout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    sigma2 = max(1e-12, float(gauss_width) ** 2)

    def compute_field(pos: np.ndarray) -> np.ndarray:
        if pos.shape[0] == 0:
            return np.zeros((grid_res, grid_res), dtype=np.float32)
        # Chunk over grid points to reduce memory
        field = np.empty(grid_pts.shape[0], dtype=np.float32)
        chunk = max(1, (grid_res * grid_res) // 16)
        for start in range(0, grid_pts.shape[0], chunk):
            gp = grid_pts[start:start+chunk]  # (M,2)
            # d2: (M, N)
            d2 = ((gp[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)
            g = np.exp(- d2 / (2.0 * sigma2))
            if mode == "sum":
                v = g.sum(axis=1)
            else:
                v = g.max(axis=1)
            field[start:start+gp.shape[0]] = v.astype(np.float32)
        # Normalize to [0,1] for display
        m = float(field.max()) if field.size > 0 else 0.0
        if m > 0.0:
            field = field / m
        return field.reshape(grid_res, grid_res)

    def init():
        im.set_data(np.zeros((grid_res, grid_res), dtype=np.float32))
        scat.set_offsets(np.empty((0, 2), dtype=np.float32))
        scat.set_array(np.array([]))
        artists = []
        if target_artist is None:
            artists = [im, scat]
        else:
            artists = [target_artist, im, scat]
        if draw_cutoff:
            # Initialize empty collection for cutoff circles
            nonlocal cutoff_artist
            cutoff_artist = PatchCollection([], facecolor='none', edgecolor='cyan', linewidths=0.7, alpha=0.4, zorder=2.5)
            cutoff_artist.set_animated(True)
            ax.add_collection(cutoff_artist)
            artists.append(cutoff_artist)
        return tuple(artists)

    def update(frame):
        pos = positions[frame] if frame < len(positions) else np.empty((0, 2), dtype=np.float32)
        w = states_t[frame][:,state_index]

        # add fitness to title
        fitness = fitness_history[frame]
        ax.set_title(f"Gaussian Field + State — t={frame+1} (n={pos.shape[0]}) Fitness: {fitness:.4f}")

        field = compute_field(pos)
        im.set_data(field)
        scat.set_offsets(pos)
        # Clamp colors to [0,1]
        scat.set_array(w)
        # Update cutoff circles if enabled
        artists = []
        if target_artist is None:
            artists = [im, scat]
        else:
            artists = [target_artist, im, scat]
        if draw_cutoff:
            # Remove previous collection and create a new one for current frame
            nonlocal cutoff_artist
            if cutoff_artist is not None:
                try:
                    cutoff_artist.remove()
                except Exception:
                    pass
            circles = [Circle((float(p[0]), float(p[1])), radius=float(cfg.cutoff)) for p in pos]
            cutoff_artist = PatchCollection(circles, facecolor='none', edgecolor='cyan', linewidths=0.7, alpha=0.4, zorder=2.5)
            cutoff_artist.set_animated(True)
            ax.add_collection(cutoff_artist)
            artists.append(cutoff_artist)
        ax.set_title(f"Gaussian Field + State — t={frame+1} (n={pos.shape[0]})")
        return tuple(artists)

    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    anim = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=1000//fps)

    out_path = out_path or os.path.join(os.path.dirname(__file__), "rollout_gaussian.mp4")
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='ProliferateAutoencode'))
            anim.save(out_path, writer=writer)
        except Exception as e:
            print(f"FFMpeg not available ({e}); falling back to GIF...")
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            out_path = gif_path
    elif ext == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
    else:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='ProliferateAutoencode'))
        anim.save(out_path, writer=writer)

    plt.close(fig)
    print(f"Saved Gaussian rollout video to {out_path}")

Gaussian = True

def main():
    steps = None
    name = "move_and_color_100"
    max_cells = None

    # Provide path to a checkpoint like checkpoints/ga_gen_50.pt
    resume_path = os.environ.get("GA_RESUME", os.path.join(os.path.dirname(__file__), "checkpoints", f"{name}.pt"))
    model, saved_cfg = load_best_from_checkpoint(resume_path, rollout_steps=steps)
    # Derive visualization cfg from saved cfg, with optional overrides
    vis_cfg = saved_cfg
    vis_cfg.rollout_steps = steps or saved_cfg.rollout_steps
    vis_cfg.max_cells = max_cells or saved_cfg.max_cells
    vis_cfg.device = "cpu"
    # Show snapshots grid
    # rollout_and_show(model, steps=steps, dt=0.1, device="cpu", interval=every, max_cells=max_cells)
    # Also export a short video
    out_path = os.environ.get("GA_VIDEO_OUT", os.path.join(os.path.dirname(__file__), "videos", f"{name}_rollout.mp4"))
    
    if not Gaussian:
        rollout_and_show_video(model, out_path=out_path, fps=2, s=6, level=0, cfg=vis_cfg)
    else:
        rollout_and_show_video_gaussian(model, out_path=out_path, fps=2, level=0, grid_res=256, mode="max", cfg=vis_cfg, draw_cutoff = True, angle=0.)

if __name__ == "__main__":
    main()
