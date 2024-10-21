from evolutionary import EvolutionaryAlgorithm
from prompt import PromptParams
from datetime import datetime
from os import getenv
import utils
from args import parse_args_and_init
import data_evaluation as eval
if __name__ == "__main__":
    ident = getenv("SLURM_JOB_ID") or datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
    log_file = f"logs/results/{ident}.ndjson"

    evo_params, splits, api = parse_args_and_init(log_file)
    train, eval_data = splits
    
    usage_handle, score_handle = utils.create_api_handles(api, log_file, evo_params.scorer)
    
    prompt_params = PromptParams(usage_handle, score_handle, log_file)
    evo_params.prompt_params = prompt_params
    EA = EvolutionaryAlgorithm(evo_params, usage_handle, train)
    EA.populate()
    EA.run()

    best_prompts = eval.best_prompts_from_each_gen(EA.all_prompts)
    generation_scores = eval.evaluate_progression(best_prompts, eval_data)
    step_scores = eval.evaluate_progression(EA.population_through_steps, eval_data)
    
    baseline_suffixes = {
        "Blank": "{}",
        "Kojima": "{}\nLet's think step by step.",  # Kojima et al. 2022
        "Zhou": "{}\nLet's work this out in a step by step way to be sure we have the right answer.",  # Zhou et al. 2022b
        "Fernando": "{}\nSOLUTION:",  # Fernando et al. 2023
    }
    
    scores = {"Generation": generation_scores, "Steps": step_scores}
    baseline_scores = {name: eval.calculate_baseline(eval_data, baseline, prompt_params) for name, baseline in baseline_suffixes.items()}
    scores.update(baseline_scores)
    
    eval.plot_generations(scores, ident)