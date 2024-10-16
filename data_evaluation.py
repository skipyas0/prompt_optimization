import utils
from datasets import Dataset
from prompt import Prompt, PromptParams
import matplotlib.pyplot as plt
from sys import argv
from prompt_templates import baseline_suffixes
def reconstruct_prompts(prompt_data: list[dict], prompt_params: PromptParams) -> list[Prompt]:
    """
    Select all dicts that represent a prompt and reconstruct them into objects
    """
    return [Prompt(p, prompt_params) for p in filter(lambda x: x['type'] == "prompt", prompt_data)]

def best_prompts_from_each_gen(prompts: list[Prompt], n: int = 3) -> list[list[Prompt]]:
    """
    Choose n prompts from each generation (k gens in total) with best avg scores.
    Outputs [k*[n*prompt]]
    """
    out = []
    gen = 1
    while len(prompts_of_gen:=list(filter(lambda x: x.generation_number == gen, prompts))) > 0:
        best_n_prompts = sorted(prompts_of_gen, key= lambda x: x.fitness, reverse=True)[:n]
        out.append(best_n_prompts)
        gen += 1
    return out

def evaluate_generation(prompts: list[Prompt], eval_data: Dataset) -> float:
    """
    Average the performance of given prompts on whole evaluation dataset
    """

    scores = [p.calculate_fitness(eval_data) for p in prompts]
    return sum(scores) / len(scores)

def evaluate_progression(prompt_generations: list[list[Prompt]], eval_data: Dataset) -> list[float]:
    return [evaluate_generation(g, eval_data) for g in prompt_generations]

def evaluate_from_json(path: str, n_per_gen: int, prompt_params: PromptParams) -> list[float]:
    """
    For later analysis, extract prompts from .ndjson, evaluate them using handles from PromptParams object.
    Output list of per-generation average scores.
    """
    json = utils.load_log_dict(path)
    prompts = reconstruct_prompts(json, prompt_params)
    best = best_prompts_from_each_gen(prompts, n_per_gen)
    return evaluate_progression(best)


def plot_generations(scores: tuple[list[float], list[float]], plot_path: str) -> None:
    by_gen, by_step = scores
    gens = range(1, len(by_gen) + 1)
    steps = range(1, len(by_step) + 1)
    # Create the plot
    plt.figure()
    plt.plot(gens, by_gen, marker='o', linestyle='-', color='b')
    plt.plot(steps, by_step, marker='o', linestyle='-', color='r')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution progress')

    # Save the plot as a vector image (SVG)
    plt.savefig(f'plots/{plot_path}.svg', format='svg')

def calculate_baseline(eval_data: Dataset, baseline_prompt: Prompt) -> float:
    pass

if __name__ == "__main__":
    from data_evaluation import evaluate_from_json
    slurm_id = argv[1]
    n_best_from_gen = 3
    scores = evaluate_from_json(f"{slurm_id}.ndjson", n_best_from_gen)