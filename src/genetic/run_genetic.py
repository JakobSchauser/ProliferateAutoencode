import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from genetic.simulation_genetic import GAConfig, genetic_train

def main():
    cfg = GAConfig(
        population_size=16,
        elite_frac=0.25,
        mutation_std=0.05,
        crossover_frac=0.5,
        generations=100,
        rollout_steps=200,
        dt=0.1,
        target_count=99,
        max_cells=1000,
        circle_weight=2.0,
        device="cpu",
        resume_path="checkpoints/genetic_emoji_15.pt",  # or None to train from scratch
        name = "genetic_emoji",
        emoji="🦎",
        N_times = 2,

    )
    best_model, hist = genetic_train(cfg)
    print("Genetic training done. Best fitness history length:", len(hist))



if __name__ == "__main__":
    main()
