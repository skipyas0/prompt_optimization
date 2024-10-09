from datasets import load_dataset, Dataset
import re
import json
from vllm_api import OpenAIPredictor
from types import NoneType
from typing import Callable, Literal
import random 

def load_log_dict(path: str) -> list[dict]:
    """
    Open .ndjson and load it as list of dicts
    """
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))  # Convert each line to a dictionary
    return data

def load_splits(ds_name: str, split: tuple[int, int, int]) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load dataset and split it according to sample numbers in input tuple.
    """
    ds = load_dataset(ds_name, 'main', split='train') if ds_name == 'openai/gsm8k' else load_dataset(ds_name, split='train')
    ds = ds.shuffle(seed=42).select(range(sum(split)))
    infer = ds.select(range(split[0]))
    train = ds.select(range(split[0], sum(split[:2])))
    test = ds.select(range(sum(split[:2]), sum(split)))
    return infer, train, test

def join_dataset_to_str(dataset: Dataset, insertion_token: str, insertion_position: Literal["prefix", "suffix"]) -> str:   
    """
    Join samples from datasets to single string with <in> <out> html-like tags.
    """

    res = ""

    features = dataset.features
    q, a = features[0], features[1]

    for i, sample in enumerate(dataset):
        res += f"<{i}>\n"
        res += insertion_token if insertion_position == "prefix" else "Follow these steps and solve the problem in a logical fashion.\n 1. Analyze the problem.\n 2. Create a plan to solve it.\n 3. Follow the plan and explain your steps.\n 4. Give your final answer.\n"
        res += f"<{q}> {sample[q]} </{q}>\n"
        res += insertion_token if insertion_position == "suffix" else "Let's think step by step.\n"
        res += f"<{a}> {sample[a]} </{a}>\n"
        res += f"</{i}>\n"

    return res

def parse_verdict(text: str) -> str:
    """
    Parse [[[verdict]]]
    """
    pattern = r'\[\[\[([a-z]+ ?[a-z]*)\]\]\]'# <- only lowercase letters 
    #any char - r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches

def parse_answer(text: str) -> str:
    """
    Parse [[answer]]
    """
    pattern = r'\[\[(.*)\]\]'
    #any char - r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches

def log_usage(log_file: str, input: str | tuple[str, str], output: str | float) -> None:
    """
    Add entry about handle usage to log file.
    """
    if type(output) == str:
        log_entry = {
            'type': "usage",
            'in': input,
            'out': output,
        }
    else:
        log_entry = {
            'type': "score",
            'ground': input[0],
            'in': input[1],
            'out': output,
        }

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def create_api_handles(api: OpenAIPredictor | NoneType, log_file: str, scorer: str) -> tuple[Callable[[str], str], Callable[[str], float]]:
    if api is None:
        #fast debug to replace LLM calls 
        def scramble(input: str, _: int = 0) -> str:
            char_list = list(input)
            random.shuffle(char_list)
            return ''.join(char_list)
        usage_helper = scramble
        score_helper = lambda _0, _1: random.random()
    else:
        import evaluators as ev
        usage_helper = lambda prompt: api.predict(question=prompt)
        if scorer == "ask_llm_to_compare":
            score_helper = lambda ground, x: ev.ask_llm_to_compare(ground, x, usage_handle)
        elif scorer == "binary_match":
            score_helper = lambda ground, x: ev.binary_match(ground, x)

    def usage_handle(input: str) -> str:
        out = usage_helper(input)
        log_usage(log_file, input, out)
        return out
    
    def score_handle(ground: str, input: str) -> float:

        # if this is the gsm8k dataset, the answer is EXPLANATION #### NUMERIC ANSWER
        spl = ground.split("####")
        if len(spl) == 2:
            ground = spl[1].strip()

        out = score_helper(ground, input)
        log_usage(log_file, (ground, input), out)
        return out
    
    return usage_handle, score_handle