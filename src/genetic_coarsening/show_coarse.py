import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from simulation_coarse import GAConfig, _load_checkpoint, evaluate_model
from NCAArchitecture import ParticleNCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from target_functions import sample_positions_from_image

def load_best_from_checkpoint(resume_path: str, rollout_steps: int = 1) -> ParticleNCA:
    cfg = GAConfig(resume_path=resume_path, rollout_steps=rollout_steps,)
    start_gen, population, history = _load_checkpoint(cfg)
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
    return population[best_idx]


# Removed rollout_and_show; use evaluate_model return for visualization


def rollout_and_show_video(model: ParticleNCA, steps: int = 300, dt: float = 0.1, device: str = "cpu", out_path: str = "rollout.mp4", fps: int = 30, s: int = 6, max_cells: int = 256, level: int = 1):
    """
    Roll out the world and render an animation. Saves to MP4 if out_path ends with .mp4,
    or GIF if it ends with .gif. If out_path is None, defaults to 'rollout.mp4'.
    """
    cfg = GAConfig(dt=dt, device=device, max_cells=max_cells)

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=s, alpha=0.8, cmap='viridis', zorder=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("NCA Rollout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Prepare target overlay points once
    target_pts = sample_positions_from_image(level)

    # Use evaluate_model to get positions and per-frame fitness for visualization
    cfg = GAConfig(dt=dt, device=device, max_cells=max_cells, rollout_steps=steps)
    res = evaluate_model(
        cfg,
        model,
        N_times=1,
        return_positions=True,
        record_interval=1,
        return_fitness_per_frame=True,
    )
    if isinstance(res, tuple):
        avg_fit, positions_t, fitnesses = res
    else:
        avg_fit = float(res)
        positions_t, fitnesses = [], []
    positions = [p.detach().cpu().numpy() for p in positions_t]
    colors = []
    prev_n = 0
    for pos in positions:
        n = pos.shape[0]
        frame_colors = np.full(n, 0.2, dtype=np.float32)
        if n > prev_n:
            frame_colors[:prev_n] = 0.2
            frame_colors[prev_n:n] = 0.95
        colors.append(frame_colors)
        prev_n = n

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
        col = colors[frame]
        scat.set_offsets(pos)
        scat.set_array(col)
        fit = fitnesses[frame] if frame < len(fitnesses) else None
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
    steps: int = 300,
    dt: float = 0.1,
    device: str = "cpu",
    out_path: str = "rollout_gaussian.mp4",
    fps: int = 30,
    max_cells: int = 256,
    level: int = 1,
    gauss_width: float = 0.1,
    grid_res: int = 256,
    mode: str = "max",
):
    """
    Render an animation of the Gaussian field induced by particle positions.
    - Field is either the max or sum of per-particle Gaussians with width `gauss_width`.
    - Overlays target points normalized to [-1,1] with inverted Y to match plot coords.
    """
    cfg = GAConfig(dt=dt, device=device, max_cells=max_cells, rollout_steps=steps)

    # Load target points and normalize to [-1, 1] for overlay
    target_np = sample_positions_from_image(level)
    target_artist = None
    if target_np is not None and target_np.size > 0:
        tp = target_np.astype(np.float32)
        xmin, ymin = tp.min(axis=0)
        xmax, ymax = tp.max(axis=0)
        w = max(1e-6, xmax - xmin)
        h = max(1e-6, ymax - ymin)
        tx = (tp[:, 0] - xmin) / w * 2.0 - 1.0
        ty = (tp[:, 1] - ymin) / h * 2.0 - 1.0
        ty = -ty  # invert Y for display
        target_pts = np.stack([tx, ty], axis=1)
    else:
        target_pts = None

    # Rollout to collect positions per frame
    res = evaluate_model(
        cfg,
        model,
        N_times=1,
        return_positions=True,
        record_interval=1,
        return_fitness_per_frame=False,
    )
    if isinstance(res, tuple):
        _, positions_t, _ = res
    else:
        positions_t = []
    positions = [p.detach().cpu().numpy() for p in positions_t]

    # Determine axis limits
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    if target_pts is not None and target_pts.shape[0] > 0:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
        valid = [p for p in positions if p.shape[0] > 0]
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

    # Build grid over current axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xs = np.linspace(xlim[0], xlim[1], grid_res, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_res, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (H*W, 2)

    # Initialize image artist
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

    # Overlay targets, if available
    if target_pts is not None and target_pts.shape[0] > 0:
        target_artist = ax.scatter(target_pts[:, 0], target_pts[:, 1], s=6, alpha=0.3, color='black', zorder=2)

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
        return (im,) if target_artist is None else (target_artist, im)

    def update(frame):
        pos = positions[frame] if frame < len(positions) else np.empty((0, 2), dtype=np.float32)
        field = compute_field(pos)
        im.set_data(field)
        ax.set_title(f"Gaussian Field — t={frame+1} (n={pos.shape[0]})")
        return (im,) if target_artist is None else (target_artist, im)

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
    steps = 200
    every = 20
    name = "tjæst_0"
    max_cells = 500

    # Provide path to a checkpoint like checkpoints/ga_gen_50.pt
    resume_path = os.environ.get("GA_RESUME", os.path.join(os.path.dirname(__file__), "checkpoints", f"{name}.pt"))
    model = load_best_from_checkpoint(resume_path, rollout_steps=steps)
    # Show snapshots grid
    # rollout_and_show(model, steps=steps, dt=0.1, device="cpu", interval=every, max_cells=max_cells)
    # Also export a short video
    out_path = os.environ.get("GA_VIDEO_OUT", os.path.join(os.path.dirname(__file__), "videos", f"{name}_rollout.mp4"))
    
    if not Gaussian:
        rollout_and_show_video(model, steps=steps, dt=0.1, device="cpu", out_path=out_path, fps=2, s=6, max_cells=max_cells, level = 0)
    else:
        rollout_and_show_video_gaussian(model, steps=steps, dt=0.1, device="cpu", out_path=out_path, fps=2, max_cells=max_cells, level = 0, gauss_width=0.1, grid_res=256, mode="max")

if __name__ == "__main__":
    main()
