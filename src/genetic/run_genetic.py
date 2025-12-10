import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from genetic.simulation_genetic import GAConfig, genetic_train

def main():
    cfg = GAConfig(
        population_size=16,
        elite_frac=0.2,
        mutation_std=0.05,
        crossover_frac=0.5,
        generations=100,
        rollout_steps=100,
        dt=0.1,
        target_count=128,
        max_cells=2000,
        device="cpu",
        resume_path=None,  # or None to train from scratch
        name = "genetic_emoji_fifth_higher_cardinality",
        emoji="🦎",
        N_times = 8,

    )
    best_model, hist = genetic_train(cfg, max_workers=4)
    print("Genetic training done. Best fitness history length:", len(hist))



if __name__ == "__main__":
    main()
