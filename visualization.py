import matplotlib.pyplot as plt
from sys import argv

def plot_generations(scores: list[float], plot_path: str) -> None:
    x = range(1, len(scores) + 1)
    # Create the plot
    plt.figure()
    plt.plot(x, scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution progress')

    # Save the plot as a vector image (SVG)
    plt.savefig(f'plots/{plot_path}.svg', format='svg')

if __name__ == "__main__":
    from reconstruct import evaluate_from_json
    slurm_id = argv[1]
    n_best_from_gen = 3
    scores = evaluate_from_json(f"{slurm_id}.ndjson", n_best_from_gen)
