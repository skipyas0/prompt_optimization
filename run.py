from evolutionary import EvolutionaryAlgorithm, EvoParams
import evaluators as eval
from prompt import Prompt, PromptParams
from vllm_api import OpenAIPredictor
from datetime import datetime

from os import getenv

if __name__ == "__main__":
    task_category = "entity_extraction"
    rel_path = f"tasks/{task_category}/"
    task_name = "german-intelligence-says-russian-gru-group-behind-nato-eu-cyberattacks-2024-09-09"

    with open(f"{rel_path}/generation.txt") as f:
        gen_instructions = f.read()
    with open(f"{rel_path}/samples/x/{task_name}.txt") as f:
        task = f.read()
    with open(f"{rel_path}/samples/y/{task_name}.txt") as f:
        ground = f.read()

    model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    #model = "mistralai/Mistral-7B-v0.3"
    #api = OpenAIPredictor(model)
    prelude = [{"role": "system", "content": "You are a helpful assistant. You follow instructions and answer concisely."}]
    #gen_handle = lambda x: api.predict(prelude, x)

    score = lambda x: eval.simple_list_intersection(ground, x)

    
    #fast debug to replace LLM calls
    import random 
    gen_instructions = "Hello world! This is a testy test."
    task = "Generic task :O. What could it be?"
    def scramble(input: str) -> str:
        char_list = list(input)
        random.shuffle(char_list)
        return ''.join(char_list)
    gen_handle = scramble
    score = lambda _: random.random()
    

    ident = getenv("SLURM_JOB_ID") or datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
    log_file = f"logs/results/{task_category}_{ident}.txt"

    # prompt has three traits - persona (0), task introduction (1) <insert task here>, generation start sequence (2)
    insert_ix = 2
    prompt_params = PromptParams(gen_handle, score, log_file, insert_ix)
    params = EvoParams(initial_population_size=5,max_iters=10, mating_pool_size=3)
    EA = EvolutionaryAlgorithm(params, gen_handle, task)
    EA.populate(prompt_params)
    EA.run()
