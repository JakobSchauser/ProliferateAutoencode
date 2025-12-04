import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from genetic.simulation_genetic import GAConfig, _load_checkpoint, ParticleWorld
from NCAArchitecture import ParticleNCA


def load_best_from_checkpoint(resume_path: str) -> ParticleNCA:
    cfg = GAConfig(resume_path=resume_path)
    start_gen, population, history = _load_checkpoint(cfg)
    # Pick first individual from loaded population
    # Alternatively, re-evaluate and pick best; for speed, just use index 0
    model = population[0]
    print(f"Loaded checkpoint gen={start_gen}, population size={len(population)}, history len={len(history)}")
    return model


def rollout_and_show(model: ParticleNCA, steps: int = 300, dt: float = 0.1, device: str = "cpu", interval: int = 10, max_cells: int = 256):
    cfg = GAConfig(dt=dt, device=device, max_cells=max_cells)
    world = ParticleWorld(cfg, model)
    world.reset(n0=1)

    xs = []
    ts = []
    for t in range(steps):
        world.step()
        if (t + 1) % interval == 0:
            if world.x is not None:
                xs.append(world.x.detach().cpu())
                ts.append(t + 1)
                print(f"Step {t+1}/{steps} | cells={world.x.shape[0]}")

    # Plot snapshots every `interval`
    n = len(xs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), squeeze=False)
    for i, (snap, tstep) in enumerate(zip(xs, ts)):
        r = i // cols
        c = i % cols
        pos = snap.numpy()
        ax = axes[r][c]
        ax.scatter(pos[:, 0], pos[:, 1], s=4, alpha=0.7)
        ax.set_title(f"t={tstep} (n={pos.shape[0]})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
    # Hide any unused axes
    for j in range(n, rows*cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.show()


def rollout_and_show_video(model: ParticleNCA, steps: int = 300, dt: float = 0.1, device: str = "cpu", out_path: str = "rollout.mp4", fps: int = 30, s: int = 6, max_cells: int = 256):
    """
    Roll out the world and render an animation. Saves to MP4 if out_path ends with .mp4,
    or GIF if it ends with .gif. If out_path is None, defaults to 'rollout.mp4'.
    """
    cfg = GAConfig(dt=dt, device=device, max_cells=max_cells)
    world = ParticleWorld(cfg, model)
    world.reset(n0=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=s, alpha=0.8, cmap='viridis')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("NCA Rollout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    positions = []
    colors = []
    prev_n = 0
    for t in range(steps):
        world.step()
        pos = world.x.detach().cpu().numpy() if world.x is not None else np.empty((0,2), dtype=np.float32)
        positions.append(pos)
        # Color newly divided cells differently (red-ish via higher colormap value)
        n = pos.shape[0]
        frame_colors = np.full(n, 0.2, dtype=np.float32)
        if n > prev_n:
            frame_colors[:prev_n] = 0.2
            frame_colors[prev_n:n] = 0.95
        colors.append(frame_colors)
        prev_n = n

    # Compute global axis limits to keep all cells in frame
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

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        return scat,

    def update(frame):
        pos = positions[frame]
        col = colors[frame]
        scat.set_offsets(pos)
        scat.set_array(col)
        ax.set_title(f"t={frame+1} (n={pos.shape[0]})")
        return scat,

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
    steps = 200
    every = 20
    name = "genetic_emoji_15"

    max_cells = 1000

    # Provide path to a checkpoint like checkpoints/ga_gen_50.pt
    resume_path = os.environ.get("GA_RESUME", os.path.join(os.path.dirname(__file__), "checkpoints", f"{name}.pt"))
    model = load_best_from_checkpoint(resume_path)
    # Show snapshots grid
    # rollout_and_show(model, steps=steps, dt=0.1, device="cpu", interval=every, max_cells=max_cells)
    # Also export a short video
    out_path = os.environ.get("GA_VIDEO_OUT", os.path.join(os.path.dirname(__file__), f"{name}_rollout.mp4"))
    rollout_and_show_video(model, steps=steps, dt=0.1, device="cpu", out_path=out_path, fps=2, s=6, max_cells=max_cells)


if __name__ == "__main__":
    main()
