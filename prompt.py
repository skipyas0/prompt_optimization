from __future__ import annotations
from typing import Callable

class PromptParams():
    def __init__(self, usage_handle: Callable[[list[str]], str], evalutation_handle: Callable[[str], float], log_file: str, task_insert_index: int) -> None:
        self.usage_handle = usage_handle
        self.evaluation_handle = evalutation_handle
        self.log_file = log_file
        self.task_insert_index = task_insert_index

class Prompt():
    def __init__(self, traits: list[str], params: PromptParams) -> None:    
        self.params = params
        self.traits = traits
        self.n_traits = len(self.traits)
        self.fitness = float('-inf')
        self.generation_number = 1
        self.result = ""

    def __str__(self) -> str:
        ix = self.params.task_insert_index
        return '\n'.join(self.traits[:ix]) + '\n<task>\n{}\n</task>\n' + '\n'.join(self.traits[ix:]) 

    def calculate_fitness(self) -> float:
        self.result = self.params.usage_handle(str(self))
        self.fitness = self.params.evaluation_handle(self.result)
        return self.fitness

    def copy(self) -> Prompt:
        new = Prompt(self.traits.copy(), self.params)
        new.generation_number = self.generation_number
        return new
    
    def log(self) -> None:
        with open(self.params.log_file, 'a') as f:
            f.write(f"Prompt gen: {self.generation_number}\n\n***\n{str(self)}\n***\nResult:\n{self.result}\n\nScore: {self.fitness}\n\n######")