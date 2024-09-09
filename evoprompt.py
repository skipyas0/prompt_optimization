from __future__ import annotations
from typing import Callable
from evolutionary import Specimen
import os


class Prompt(Specimen):
    def __init__(self, LLM_generate_handle: Callable[[str], str], prompt: str) -> None:
        super().__init__([prompt])
        self.genetic_operator_instructions: dict[str, str] = dict()
        self.load_instructions()
        self.generate = LLM_generate_handle

    def load_instructions(self) -> None:
        """
        Access prewritten instructions to be used in genetic operators.
        """
        curr_path = os.getcwd()
        path = curr_path + '/genop_prompts/'
        for fn in os.listdir(path):
            op_name = fn.split('.')[0]
            with open(path + fn, 'r') as file:
                self.genetic_operator_instructions[op_name] = file.read()
        assert len(self.genetic_operator_instructions.keys()) == len(os.listdir(path))
    
    def _trait_addition(self, trait_ix: int, other: Prompt) -> None:
        """ 
        Other prompt is a string with comma-separated words. 
        This method tries to encorporate their meaning into this prompt.
        """
        self.traits[trait_ix] = self.generate(
            self.genetic_operator_instructions['addition'].format(
                self.traits[trait_ix], other.traits[trait_ix])
        )
    
    def _trait_crossover(self, trait_ix: int, other: Prompt) -> None:
        """
        Combine the other prompt into this prompt, trying to capture meaning from both of them.
        """
        self.traits[trait_ix] = self.generate(
            self.genetic_operator_instructions['crossover'].format(
                self.traits[trait_ix], other.traits[trait_ix])
        )
    
    def _trait_de_combination(self, trait_ix: int, other1: Prompt, other2: Prompt) -> None:
        """
        Identifies parts where the prompts differ and mutates them.
        This prompt is then just a space-separated list of words.
        """
        diffs = self.gen(
            self.genetic_operator_instructions['de1'].format(
                self.traits[trait_ix], other1.traits[trait_ix])
        )

        mutated = self.gen(
            self.genetic_operator_instructions['de2'].format(diffs))
        
        self.traits[trait_ix] = self.gen(
            self.genetic_operator_instructions['de3'].format(
                mutated, other2.traits[trait_ix])
        )
    
    def _trait_mutate(self, trait_ix: int) -> None:
        """
        Change prompt while conserving semantic meaning.
        """
        self.traits[trait_ix] = self.generate(
            self.genetic_operator_instructions['mutation'].format(self.traits[trait_ix])
        )
    