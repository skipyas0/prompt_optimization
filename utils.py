from datasets import Dataset
import re
import json
from typing import Literal, Type, TypeVar
import random 
import shutil
import os
import io
from contextlib import redirect_stdout

def scramble(input: str, *_, **__) -> str:
    return input
    #char_list = list(input)
    #random.shuffle(char_list)
    #return ''.join(char_list)

def load_log_dict(path: str) -> list[dict]:
    """
    Open .ndjson and load it as list of dicts
    """
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))  # Convert each line to a dictionary
    return data

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

def find_key_by_value(d, x):
    for key, value in d.items():
        if value == x:
            return key
    return None  

def copy_contents(source_folder, dest_folder):
    if not os.path.exists(dest_folder):
        raise NotADirectoryError(f"Check if the run number {dest_folder.split('/')[1]} has a directory")
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(dest_folder, item)
        
        # Check if it's a directory and copy accordingly
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)  # dirs_exist_ok=True will overwrite if exists
        else:
            shutil.copy2(source_path, destination_path)


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
    
class LoggingWrapper:
    def __init__(self, ident) -> None:
        self.ident = ident
        self.log_file = "runs/ident/results/{}.ndjson"

    def __call__(self, input, output, log_type) -> None:
        """
        Add entry about handle usage to log file.
        """

        if log_type == "usage":
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
    
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


T = TypeVar('T', bound='FromJSON')
class FromJSON:
    default_path: str = "./{}.json"
    def to_json(self, template: str) -> None:
        """Save the current instance attributes to a JSON file."""
        filepath = self.default_path.format(template)
        with open(filepath, 'w') as file:
            json.dump(self.__dict__, file, indent=4)

    @classmethod
    def from_json(cls: Type[T], template: str) -> T:
        """Load configuration from a JSON file and return an instance of the class."""
        filepath = cls.default_path.format(template)
        with open(filepath, 'r') as file:
            config = json.load(file)
        return cls(name=template, **config)
    
class MyCollectionIterator:
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __next__(self):
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration
