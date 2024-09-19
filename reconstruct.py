import utils
from datasets import Dataset
from prompt import Prompt, PromptParams


def reconstruct_prompts(prompt_data: list[dict], prompt_params: PromptParams) -> list[Prompt]:
    return [Prompt(p, prompt_params) for p in prompt_data]

def select_best_prompts(prompts: list[Prompt], n: int) -> list[list[Prompt]]:
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

def evaluate_from_json(path: str, n_per_gen: int) -> list[float]:
    json = utils.load_log_dict(path)
    prompts = reconstruct_prompts(json)
    best = select_best_prompts(prompts, n_per_gen)
    return evaluate_progression(best)