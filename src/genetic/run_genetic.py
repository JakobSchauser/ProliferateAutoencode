import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from genetic.simulation_genetic import GAConfig, genetic_train

def main():
    cfg = GAConfig(
        population_size=32,
        elite_frac=0.2,
        mutation_std=0.001,
        generations=1600,
        rollout_steps=100,
        dt=0.1,
        target_count=128,
        max_cells=500,
        device="cpu",
        resume_path=None,  #"checkpoints/genetic_emoji_seventh_60.pt",  # or None to train from scratch
        name = "genetic_emoji_eighth",
        emoji="🦎",
        N_times = 16,

    )
    best_model, hist = genetic_train(cfg, use_threads=False, max_workers=16)
    print("Genetic training done. Best fitness history length:", len(hist))



if __name__ == "__main__":
    main()
