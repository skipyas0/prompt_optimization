from evolutionary import EvolutionaryAlgorithm
import evaluators as eval
from prompt import PromptParams
from datetime import datetime
from os import getenv
import utils
from args import parse_args_and_init

if __name__ == "__main__":
    evo_params, splits, api = parse_args_and_init()
    infer, train, test = splits
    examples = utils.join_dataset_to_str(infer)

    usage_handle, score_handle = utils.create_api_handles(api)
    

    ident = getenv("SLURM_JOB_ID") or datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
    log_file = f"logs/results/{ident}.ndjson"

    prompt_params = PromptParams(usage_handle, score_handle, log_file, evo_params.task_insert_ix)
    EA = EvolutionaryAlgorithm(evo_params, usage_handle, train)
    EA.populate(prompt_params, examples)
    EA.run()
