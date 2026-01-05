import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation_coarse import GAConfig, genetic_train

def main():
    cfg = GAConfig(
        population_size=24,
        elite_frac=0.2,
        mutation_std=0.001,
        generations=500,
        rollout_steps=200,
        dt=0.1,
        target_count=128,
        max_cells=500,
        device="cpu",
        cutoff=0.05,
        cell_size = 0.05,
        resume_path=None,#"checkpoints/move_and_color_100.pt",  # or None to train from scratch
        name = "three_test",
        N_times = 20,
        n_molecules = 8,
    )
    best_model, hist = genetic_train(cfg, use_threads=False, max_workers=16, random_start_angle=False)
    print("Genetic training done. Best fitness history length:", len(hist))



if __name__ == "__main__":
    main()
