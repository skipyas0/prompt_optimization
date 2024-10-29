import utils
from datasets import Dataset
from prompt import Prompt, PromptParams, Trait
import matplotlib.pyplot as plt
from sys import argv
import itertools
from stats import stats
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


def plot_generations(scores: dict[str, list[float]], ident: str, file_name:str) -> None:
    plt.figure()
    color_cycle = itertools.cycle(plt.cm.get_cmap('tab10').colors)

    for name, values in scores.items():
        color = next(color_cycle)
        generations = range(1, len(values) + 1) 
        if len(values) == 1:
            plt.axhline(y=values[0], color=color, label=name, linestyle='-.', alpha=0.5)
        else:
            plt.plot(generations, values, label=name, color=color)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution progress')
    
    plt.legend()  
    plt.savefig(f'runs/{ident}/plots/{file_name}.svg', format='svg')

def plot_training_stats(ident: str) -> None:
    color_cycle = itertools.cycle(plt.cm.get_cmap('tab10').colors)

    for name, values in stats.get_averages().items():
        if type(values) != list:
            continue
        plt.figure()
        color = next(color_cycle)
        generations = range(1, len(values) + 1) 
        plt.plot(generations, values, label=name, color=color)

        plt.ylabel(f'{name}')
        plt.xlabel('Step')
        plt.title(f'Progress of {name.lower()} in training')
        
        plt.legend()
        plt.savefig(f'runs/{ident}/plots/{name.lower().replace(" ", "_")}.svg', format='svg')

def calculate_baseline(eval_data: Dataset, baseline_prompt: Prompt, prompt_params: PromptParams) -> list[float]:
    """
    Evaluate model performace on dataset using a baseline suffix prompt, such as "Let's think step by step."
    """
    trait = Trait(baseline_prompt, 'suffix', False)
    prompt = Prompt([trait], prompt_params)
    return [prompt.calculate_fitness(eval_data)]

if __name__ == "__main__":
    slurm_id = argv[1]
    n_best_from_gen = 3
    scores = evaluate_from_json(f"{slurm_id}.ndjson", n_best_from_gen)