from datasets import Dataset
import re
import json
from typing import Literal, Type, TypeVar
import random 
import io
from contextlib import redirect_stdout
import numpy as np
from collections import namedtuple

def scramble(input: str, *_, **__) -> str:
    char_list = list(input)
    random.shuffle(char_list)
    return ''.join(char_list)

def my_exec(input_data, code, queue):
    f = io.StringIO()
    inp = iter(input_data)
    local_vars = {"input": lambda: next(inp)}
    with redirect_stdout(f):
        exec(code, {}, local_vars)
    # Send the captured output back to the main process
    queue.put(f.getvalue().strip())

def join_dataset_to_str(dataset: Dataset, insertion_token: str = "<-INS->\n", insertion_position: Literal["prefix", "suffix"] = "prefix") -> list[str]:   
    """
    Modify samples with <in> <out> html-like tags
    """

    res = []

    features = list(dataset.features.keys())
    q, a = features[0], features[1]

    for i, sample in enumerate(dataset.shuffle()):
        ex = ""
        ex += f"<{i}>\n"
        ex += f"{insertion_token}<{q}> {sample[q]} </{q}>\n" if insertion_position == "prefix" else f"<{q}> {sample[q]} </{q}>\n{insertion_token}"
        ex += f"<{a}> {sample[a]} </{a}>\n"
        ex += f"</{i}>\n"
        res.append(ex)
    return res

def parse_answer(text: str) -> str:
    """
    Parse [[answer]]
    """
    pattern = r'\[\[(.*)\]\]'
    #any char - r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches


stats = namedtuple("Stats", ["mean", "median", "min", "max"])
def seq_stats(data: list[float]):
    mean = np.mean(data)
    median = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return stats(mean, median, min_val, max_val)

class DotDict(dict):
    """A dictionary that supports dot notation access."""
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'DotDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self) -> dict:
        d = dict()
        d.update(self)
        return d

T = TypeVar('T', bound='FromJSON')
class FromJSON:
    default_path: str = "./{}.json"
    def to_json(self, target: str) -> None:
        """Save the current instance attributes to a JSON file."""
        filepath = self.default_path.format(target)
        with open(filepath, 'w') as file:
            json.dump(self.__dict__, file, indent=4)

    @classmethod
    def from_json(cls: Type[T], template: str) -> T:
        """Load configuration from a JSON file and return an instance of the class."""
        filepath = cls.default_path.format(template)
        with open(filepath, 'r') as file:
            config = json.load(file)
        return cls(name=template, **config)
    
