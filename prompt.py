from __future__ import annotations
from typing import Callable

class PromptUtils():
    def __init__(self, usage_handle: Callable[[list[str]], str], evalutation_handle: Callable[[str], float], log_file: str) -> None:
        self.usage_handle = usage_handle
        self.evaluation_handle = evalutation_handle
        self.log_file = log_file

class Prompt():
    def __init__(self, traits: list[str], utils: PromptUtils) -> None:    
        self.utils = utils
        self.traits = traits
        self.n_traits = len(self.traits)
        self.fitness = float('-inf')
        self.generation_number = 1
        self.result = ""

    def calculate_fitness(self) -> float:
        self.result = self.utils.usage_handle(self.traits)
        self.fitness = self.utils.evaluation_handle(self.result)
        return self.fitness

    def copy(self) -> Prompt:
        new = Prompt(self.traits.copy())
        new.generation_number = self.generation_number
        return new
    
    def log(self) -> None:
        with open(self.utils.log_file, 'a') as f:
            f.write(f"Prompt gen: {self.generation_number}\n\n***\n{'\n'.join(self.traits)}\n***\nResult:\n{self.result}\n\nScore: {self.fitness}\n\n######")