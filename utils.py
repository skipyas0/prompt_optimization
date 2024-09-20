from datasets import load_dataset, Dataset
import re
import json
from vllm_api import OpenAIPredictor
from types import NoneType
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

def join_dataset_to_str(dataset: Dataset) -> str:   
    """
    Join samples from datasets to single string with <in> <out> html-like tags.
    """

    res = ""

    features = dataset.features
    for sample in dataset:
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

def create_api_handles(api: OpenAIPredictor | NoneType):
    if api is None:
        #fast debug to replace LLM calls 
        def scramble(input: str, _: int = 0) -> str:
            char_list = list(input)
            random.shuffle(char_list)
            return ''.join(char_list)
        usage_handle = scramble
        score = lambda _0, _1: random.random()
    else:
        import evaluators as eval
        usage_handle = lambda prompt: api.predict(question=prompt)
        score = lambda ground, x: eval.ask_llm_to_compare(ground, x, usage_handle)
    return usage_handle, score