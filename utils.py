from datasets import load_dataset, Dataset
import re
import json
from vllm_api import OpenAIPredictor
from types import NoneType
from typing import Callable
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
    ds = load_dataset(ds_name, split='train').shuffle(seed=42).select(range(sum(split)))
    infer = ds.select(range(split[0]))
    train = ds.select(range(split[0], sum(split[:2])))
    test = ds.select(range(sum(split[:2]), sum(split)))
    return infer, train, test

def join_dataset_to_str(dataset: Dataset, insertion_token: str) -> str:   
    """
    Join samples from datasets to single string with <in> <out> html-like tags.
    """

    res = ""

    features = dataset.features
    for sample in dataset:
        res += insertion_token
        for feature in features:
            res += f"<{feature}> {sample[feature]} </{feature}>\n"
        res += '\n'

    return res

def parse_verdict(text: str) -> str:
    """
    Parse [[[verdict]]]
    """
    pattern = r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches

def log_usage(log_file: str, input: str, output: str | float) -> None:
    """
    Add entry about handle usage to log file.
    """
    log_entry = {
        'type': "usage" if type(output) == str else "score",
        'in': input,
        'out': output,
    }
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def create_api_handles(api: OpenAIPredictor | NoneType, log_file: str) -> tuple[Callable[[str], str], Callable[[str], float]]:
    if api is None:
        #fast debug to replace LLM calls 
        def scramble(input: str, _: int = 0) -> str:
            char_list = list(input)
            random.shuffle(char_list)
            return ''.join(char_list)
        usage_helper = scramble
        score_helper = lambda _: random.random()
    else:
        import evaluators as eval
        usage_helper = lambda prompt: api.predict(question=prompt)
        score_helper = lambda ground, x: eval.ask_llm_to_compare(ground, x, usage_handle)

    def usage_handle(input: str) -> str:
        out = usage_helper(input)
        log_usage(log_file, input, out)
        return out
    
    def score_handle(input: str) -> float:
        out = score_helper(input)
        log_usage(log_file, input, out)
        return out
    
    return usage_handle, score_handle