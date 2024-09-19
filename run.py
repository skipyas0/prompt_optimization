from evolutionary import EvolutionaryAlgorithm, EvoParams
import evaluators as eval
from prompt import PromptParams
from vllm_api import OpenAIPredictor
from datetime import datetime
from os import getenv
import utils

if __name__ == "__main__":
    ds_name = "microsoft/orca-math-word-problems-200k"
    split = [4,100,30]
    infer, train, test = utils.load_splits(ds_name, split)
    print("Dataset loaded:",ds_name)

    examples = utils.join_dataset_to_str(infer)

    
    model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    api = OpenAIPredictor(model)
    gen_handle_variable_length = lambda prompt, tok: api.predict(question=prompt, tok=tok)

    # TODO: Rethink how prompts access generation mainly during evaluation 
    # Now it can only optimize over one task! 
    usage_handle = lambda prompt: api.predict(question=prompt)
    score = lambda ground, x: eval.ask_llm_to_compare(ground, x, usage_handle)

    """
    #fast debug to replace LLM calls - comment out line 19
    import random 
    def scramble(input: str, _: int = 0) -> str:
        char_list = list(input)
        random.shuffle(char_list)
        return ''.join(char_list)
    gen_handle_variable_length = scramble
    usage_handle = scramble
    score = lambda _0, _1: random.random()
    """

    ident = getenv("SLURM_JOB_ID") or datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
    log_file = f"logs/results/{ident}.ndjson"

    # Specify parts of the evolved prompt and where to insert task
    trait_ids = ["instructions"]
    insert_ix = 1

    prompt_params = PromptParams(usage_handle, score, log_file, insert_ix)
    params = EvoParams(initial_population_size=10, max_iters=20, mating_pool_size=6)
    EA = EvolutionaryAlgorithm(params, gen_handle_variable_length, train)
    EA.populate(prompt_params, trait_ids, examples)
    EA.run()
