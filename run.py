from evolutionary import EvolutionaryAlgorithm
from prompt import PromptParams
from datetime import datetime
from os import getenv, mkdir
import utils
from args import parse_args_and_init
import data_evaluation as eval
from time import time
from stats import stats

if __name__ == "__main__":
    t0 = time()
    ident = getenv("SLURM_JOB_ID") or datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
    mkdir(f"runs/{ident}")
    mkdir(f"runs/{ident}/plots")
    mkdir(f"runs/{ident}/steps")
    log_file_handles = f"runs/{ident}/results.ndjson"
    log_file_prompts = f"runs/{ident}/steps/" + "step{}.ndjson"
    suff, evo_params, splits, api = parse_args_and_init(ident)
    train, eval_data = splits
    usage_handle, score_handle = utils.create_api_handles(api, log_file_handles, evo_params.scorer)
    usage_EA = lambda prompt: usage_handle(prompt, evo_params.temp)
    usage_solve = lambda prompt: usage_handle(prompt, evo_params.sol_temp)
    prompt_params = PromptParams(usage_solve, score_handle, log_file_prompts, suff)
    evo_params.prompt_params = prompt_params
    EA = EvolutionaryAlgorithm(evo_params, usage_EA, train)
    print("Init done")
    t1 = time()
    if len(evo_params.continue_run) > 0:
        EA.load_population(evo_params.continue_run)
    else: 
        EA.populate()
    t2 = time()
    EA.run()
    t3 = time()
    print("Run complete, starting eval")
    best_prompts = eval.best_prompts_from_each_gen(EA.all_prompts)
    generation_scores = eval.evaluate_progression(best_prompts, eval_data)
    step_scores = eval.evaluate_progression(EA.population_through_steps, eval_data)
    
    baseline_suffixes = {
        "Blank": "{}",
        "Kojima": "{}\nLet's think step by step.",  # Kojima et al. 2022
        #"Zhou": "{}\nLet's work this out in a step by step way to be sure we have the right answer.",  # Zhou et al. 2022b
        #"Fernando": "{}\nSOLUTION:",  # Fernando et al. 2023
    }
    
    scores_gen = {"Generation": generation_scores}
    scores_steps= {"Steps": step_scores}
    baseline_scores = {name: eval.calculate_baseline(eval_data, baseline, prompt_params) for name, baseline in baseline_suffixes.items()}
    scores_gen.update(baseline_scores)
    scores_steps.update(baseline_scores)
    
    
    eval.plot_generations(scores_gen, ident, "generations")
    eval.plot_generations(scores_steps, ident, "steps")
    eval.plot_training_stats(ident)
    t4 = time()

    stats.set_const_stat({
        "Init time": t1 - t0,
        "Populate time": t2 - t1,
        "Run time": t3 - t2,
        "Eval time": t4 - t3,
        "Total time": t4 - t0
    })

    stats.dump(ident)