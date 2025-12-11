import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from genetic.simulation_genetic import GAConfig, _load_checkpoint, evaluate_model
from genetic.target_functions import load_emoji, from_image_to_postions
from NCAArchitecture import ParticleNCA
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def rollout_and_show_video(model: ParticleNCA, steps: int = 300, dt: float = 0.1, device: str = "cpu", out_path: str = "rollout.mp4", fps: int = 30, s: int = 6, max_cells: int = 256, target_emoji: str | None = None):
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
    target_pts = None
    if target_emoji is not None:
        try:
            img = load_emoji(target_emoji)
            target_pts = from_image_to_postions(img)
        except Exception as e:
            print(f"[Warn] Failed to load target emoji '{target_emoji}': {e}")
    target_artist = None

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




def main():
    steps = 100
    every = 20
    name = "genetic_emoji_sixth_50"


    max_cells = 1000

    # Provide path to a checkpoint like checkpoints/ga_gen_50.pt
    resume_path = os.environ.get("GA_RESUME", os.path.join(os.path.dirname(__file__), "checkpoints", f"{name}.pt"))
    model = load_best_from_checkpoint(resume_path, rollout_steps=steps)
    # Show snapshots grid
    # rollout_and_show(model, steps=steps, dt=0.1, device="cpu", interval=every, max_cells=max_cells)
    # Also export a short video
    out_path = os.environ.get("GA_VIDEO_OUT", os.path.join(os.path.dirname(__file__), f"{name}_rollout.mp4"))
    # Overlay target emoji; set to None to disable overlay
    target_emoji = os.environ.get("GA_TARGET_EMOJI", "🦎")
    target_emoji = None if (target_emoji is not None and target_emoji.lower() == "none") else target_emoji
    rollout_and_show_video(model, steps=steps, dt=0.1, device="cpu", out_path=out_path, fps=2, s=6, max_cells=max_cells, target_emoji=target_emoji)


if __name__ == "__main__":
    main()
