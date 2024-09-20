from evolutionary import EvolutionaryAlgorithm
from prompt import PromptParams
from datetime import datetime
from os import getenv
import utils
from args import parse_args_and_init
from reconstruct import best_prompts_from_each_gen, evaluate_progression
from visualization import plot_generations

if __name__ == "__main__":
    ident = getenv("SLURM_JOB_ID") or datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
    log_file = f"logs/results/{ident}.ndjson"

    evo_params, splits, api = parse_args_and_init(log_file)
    infer, train, eval_data = splits

    instruction_insertion_token = "<-INS->\n"
    examples = utils.join_dataset_to_str(infer, instruction_insertion_token)
    
    usage_handle, score_handle = utils.create_api_handles(api, log_file)
    
    suffix = """
    After your explanation, make sure you put your final answer in three pairs of square brackets.
    <example>
    ...
    And for the above reasons, the solution is ANSWER.
    [[[ANSWER]]]
    </example>
    """
    prompt_params = PromptParams(usage_handle, score_handle, log_file, evo_params.task_insert_ix, suffix)
    EA = EvolutionaryAlgorithm(evo_params, usage_handle, train)
    EA.populate(prompt_params, examples)
    EA.run()

    best_prompts = best_prompts_from_each_gen(EA.all_prompts)
    generation_scores = evaluate_progression(best_prompts, eval_data)
    plot_generations(generation_scores, ident)