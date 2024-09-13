from __future__ import annotations
from random import shuffle, random, sample
from typing import Literal, Callable
from prompt import Prompt
import selection_mechanisms as sm
import genetic_operators as op
import os

class EvoParams(): 
    """
    Collection of parameters for evolutionary algorithm
    """
    def __init__(self,
                 initial_population_size: int=20,
                 population_change_rate: int=0,
                 mating_pool_size: int=10,
                 prompt_trait_n: int=1,
                 prompt_mutation_probability: float=0.2,
                 trait_mutation_percentage: float=1.0,
                 max_iters: int=100,
                 evolution_mode: Literal['GA','DE']='GA',
                 selection_mode: Literal['rank', 'roulette', 'tournament']='rank',
                 tournament_group_size: int=3)-> None:
        self.initial_population_size = initial_population_size
        self.population_change_rate = population_change_rate
        self.mating_pool_size = mating_pool_size
        assert self.initial_population_size >= self.mating_pool_size
        self.prompt_trait_n = prompt_trait_n
        self.prompt_mutation_probability = prompt_mutation_probability
        self.trait_mutation_percentage = trait_mutation_percentage
        self.max_iters = max_iters
        self.evolution_mode = evolution_mode
        self.selection_mode = selection_mode
        self.tournament_group_size = tournament_group_size

class EvolutionaryAlgorithm():
    """
    Class for prompt optimalization using evolutionary algorithms.
    """

    def __init__(self, params: EvoParams, generation_handle: Callable[[str], str]) -> None:
        self.population: list[Prompt] = []
        self.params = params
        self.pop_size = self.params.initial_population_size
        self.task_specific_handles: dict[str, Callable[[list[str]], str]] = dict()
        self.gen = generation_handle
        self.load_instructions()

        self.step = self.ga_step if self.params.evolution_mode=='GA' else self.de_step

    def run(self) -> None:
        """
        Run EvolutionaryAlgorithm to max iterations.
        """
        for i in range(self.params.max_iters):
            for s in self.population:
                s.generation_number = i+1
            self.step()
    
    def ga_step(self) -> None:
        """
        One step of Genetic Algorithm.
        """
        self.population = self.selection()
        
        # GA crossover procedure
        lotto = range(self.params.mating_pool_size)
        offsprings = []
        while len(self.population) + len(offsprings) < self.pop_size:
            i1, i2 = sample(lotto, 2)
            s1, s2 = self.population[i1], self.population[i2]

            res = op.crossover(s1, s2, self.task_specific_handles['crossover'])
            offsprings.append(res)

        self.population += offsprings

        self.mutate_group(self.population)

        self.pop_size = max(self.params.mating_pool_size,
                            self.pop_size + self.params.population_change_rate)

    def de_step(self) -> None:
        """
        One step of Differential Evolution.
        """
        selected_prompts = self.selection()
        
        # DE crossover procedure
        basic = selected_prompts[-1]
        lotto = range(self.params.mating_pool_size - 1)
        offsprings = []
        while len(self.population) + len(offsprings) < self.pop_size:
            # New offspring is Crossover(Mutate(s1 - s2) + s3, basic)
            i1, i2, i3 = sample(lotto, 3)
            prompts = (self.population[i1], self.population[i2], self.population[i3])
            handles = tuple(self.task_specific_handles[s] for s in ['de1', 'de2', 'de3'])

            res1 = op.de_combination(prompts, handles)
            res2 = op.crossover(res1, basic)

            offsprings.append(res2)

        self.population += offsprings

        # shuffle to make sure a random basic prompt is being chosen
        shuffle(self.population)
        self.pop_size = max(self.params.mating_pool_size,
                            self.pop_size + self.params.population_change_rate)

    def selection(self) -> list[Prompt]:
        """
        Perform given selection type to get new mating pool.
        """
        for s in self.population:
            s.calculate_fitness()

        m = self.params.selection_mode

        if m == 'rank':
            return sm.rank_selection(self.population)
        elif m == 'roulette':
            return sm.roulette_selection(self.population)
        else:
            return sm.tournament_selection(self.population)
    
    def mutate_group(self, to_be_mutated: list[Prompt]) -> None:
        """
        In-place mutation of given prompts according to mutation probability params.
        """
        for prompt in to_be_mutated:
            if random() < self.params.prompt_mutation_probability:
                op.mutate(prompt)
                
    def load_instructions(self) -> None:
        """
        Access prewritten instructions to be used in genetic operators.
        """
        curr_path = os.getcwd()
        path = curr_path + '/genop_prompts/'
        for fn in os.listdir(path):
            op_name = fn.split('.')[0]
            with open(path + fn, 'r') as file:
                instructions = file.read()

                # Store a handle to be used with 1 or 2 prompt-traits
                # Unpack them and format them into the instruction template
                self.task_specific_handles[op_name] = lambda s: self.gen(instructions.format(*s))

        assert len(self.task_specific_handles.keys()) == len(os.listdir(path))