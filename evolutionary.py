from random import shuffle, random, sample
from math import ceil
from __future__ import annotations
from typing import Literal, TypeVar, Generic

"""
TODO:
Test
"""

"""
This module includes template classses for genetic algorithms.
Specific behaviour is achieved by custom implementation of genetic operator supporting methods.

ToBeImplemented:
Specimen._trait_mutate (GA and DE)
Specimen._trait_crossover (GA and DE)
Specimen._trait_difference (DE only)
Specimen._trait_addition  (DE only)
Specimen._fitness (GA and DE)
Specimen._generate_traits (GA and DE)
"""


class EvoParams(): 
    """
    Collection of parameters for evolutionary algorithm
    """
    def __init__(self,
                 initial_population_size: int=20,
                 population_change_rate: int=0,
                 mating_pool_size: int=10,
                 specimen_trait_n: int=1,
                 specimen_mutation_probability: float=0.2,
                 trait_mutation_percentage: float=1.0,
                 max_iters: int=100,
                 evolution_mode: Literal['GA','DE']='GA',
                 selection_mode: Literal['rank', 'roulette', 'tournament']='roulette',
                 tournament_group_size: int=3) -> None:
        self.initial_population_size = initial_population_size
        self.population_change_rate = population_change_rate
        self.mating_pool_size = mating_pool_size
        self.specimen_trait_n = specimen_trait_n
        self.specimen_mutation_probability = specimen_mutation_probability
        self.trait_mutation_percentage = trait_mutation_percentage
        self.max_iters = max_iters
        self.evolution_mode = evolution_mode
        self.selection_mode = selection_mode
        self.tournament_group_size = tournament_group_size

T = TypeVar('T')
class Specimen():
    """
    Bundle of traits to be evaluated and optimized via evolution in a evolutionary algorithm.
    Template class - some methods need to be implemented to fit use case
    """
    def __init__(self, traits: list[Generic[T]] | int=1) -> None:
        if type(traits) is int:
            self._generate_traits(traits)
        else:
            self.traits = traits
        self.n_traits = len(self.traits)
        self.fitness = float('-inf')

    def _trait_mutate(self, trait_ix: int) -> None:
        """
        In-place mutation of trait of given number.
        """
        raise NotImplementedError
    
    def mutate(self, trait_mutation_percentage: float) -> None:
        """
        In-place mutation of specimen trait-by-trait according to self.mutate_trait.
        """
        n_to_mutate = ceil(trait_mutation_percentage * self.n_traits)
        indices_to_mutate = shuffle(list(range(self.n_traits)))[:n_to_mutate]
        for ix in indices_to_mutate:
            self._trait_mutate(ix)

    def _trait_crossover(self, trait_ix: int, other: Specimen) -> None:
        """
        In-place crossover of traits from this and other specimen given trait index.
        """
        raise NotImplementedError
    
    def crossover(self, other: Specimen) -> Specimen:
        """
        Create an offspring by combining this specimen's traits with other specimen 
        trait-by-trait according to self._trait_crossover
        """
        assert self.n_traits == other.n_traits
        new = Specimen(self.traits.copy())
        for ix in range(self.n_traits):
            new._trait_crossover(ix, other)
        return new

    def _trait_de_combination(self, trait_ix: int, other1: Specimen, other2: Specimen) -> None:
        """
        In-place difference of traits from this and other specimen given trait index.
        """
        raise NotImplementedError
    
    def de_combination(self, other1: Specimen, other2: Specimen) -> Specimen:
        """
        Create a new specimen representing the difference of this specimen and the other. 
        Differentiate trait-by-trait according to self._trait_difference.
        """
        assert self.n_traits == other1.n_traits and self.n_traits == other2.n_traits
        new = Specimen(self.traits.copy())
        for ix in range(self.n_traits):
            new._trait_de_combination(ix, other1, other2)
        return new
    
    def calculate_fitness(self) -> float:
        """
        Recalculates self.fitness and returns it
        """
        self.fitness = self._fitness()
        return self.fitness

    def _fitness(self) -> float:
        """
        Calculate fitness score for this specimen in the given invironment
        """
        raise NotImplementedError
    
    def _generate_traits(self) -> None:
        """
        To be called when a blank specimen is created to fill its traits.
        """
        raise NotImplementedError
    
class EvolutionaryAlgorithm():
    """
    Universal framework class for evolutionary algorithms for optimization.
    """
    def __init__(self, params: EvoParams) -> None:
        self.population: list[Specimen] = []
        self.pop_size = self.params.initial_population_size
        self.params = params

    def populate(self, traits: list[list[Generic[T]]] | None=None) -> None:
        """
        Generate initial population of specimens either with given traits or via Specimen._generate_traits.
        """
        if traits is None:
            for _ in range(self.params.population_size):
                s = Specimen(self.params.specimen_trait_n)
                self.population.append(s)
        else:
            for ix in range(self.params.population_size):
                s = Specimen(traits[ix])
                self.population.append(s)

    def run(self) -> None:
        """
        Run EvolutionaryAlgorithm to max iterations.
        """
        for _ in range(self.params.max_iters):
            self.step()

    def step(self) -> None:
        """
        One step of evolutionary optimalization.
        """
        m = self.params.evolution_mode
        if m == 'GA':
            self.ga_step()
        elif m == 'DE':
            self.de_step()
        else:
            raise NotImplementedError("Unknown evolution mode")
        
        # Change pop size according to params but not below mating pool size
        self.pop_size = max(self.params.mating_pool_size,
                            self.pop_size + self.params.population_change_rate)

    
    def ga_step(self) -> None:
        """
        One step of Genetic Algorithm.
        """
        self.population = self.selection()
        
        # GA crossover procedure
        lotto = range(self.params.mating_pool_size)
        offsprings = []
        while len(self.population) < self.pop_size:
            i1, i2 = sample(lotto, 2)
            s1, s2 = self.population[i1], self.population[i2]
            offsprings.append(
                s1.crossover(s2)
            )

        self.population += offsprings

        self.mutate_group(self.population)

    def de_step(self) -> None:
        """
        One step of Differential Evolution.
        """
        selected_specimens = self.selection()
        
        # DE crossover procedure
        basic = selected_specimens[-1]
        lotto = range(self.params.mating_pool_size - 1)
        offsprings = []
        while len(self.population) < self.pop_size:
            # New offspring is Crossover(Mutate(s1 - s2) + s3)
            i1, i2, i3 = sample(lotto, 3)
            s1, s2, s3 = self.population[i1], self.population[i2], self.population[i3]
            s1.de_combination(s2, s3)

            offsprings.append(
                basic.crossover(s1))
        self.population += offsprings

        # shuffle to make sure a random basic specimen is being chosen
        shuffle(self.population)

    def update_population_fitness(self) -> None:
        """
        Calculate and set new fitness scores for all specimens in the population.
        """
        for s in self.population:
            s.calculate_fitness()

    def rank_selection(self) -> list[Specimen]:
        """
        Individuals are ranked based on their fitness, and selection is done based on rank.
        """
        self.update_population_fitness()

        mating_pool = sorted(self.population, 
                     key=lambda s: s.fitness, 
                     reverse=True)[:self.params.mating_pool_size]
        
        return mating_pool
        
    
    def roulette_selection(self) -> list[Specimen]:
        """
        Individuals are selected with a probability proportional to their fitness. 
        The better the fitness, the higher the probability of being selected.
        """
        
        self.update_population_fitness()

        mating_pool = sample(self.population, self.pop_size - self.params.mating_pool_size,
                             counts=[ceil(s.fitness) for s in self.population])
        
        return mating_pool

    def tournament_selection(self) -> list[Specimen]:
        """
        A group of individuals (a tournament) is randomly chosen from the population. 
        The best individual in this group is selected. 
        This is repeated until the desired number of individuals is chosen.
        """
        self.update_population_fitness()

        mating_pool = []
        for _ in range(self.params.mating_pool_size):
            group = sample(self.population, self.params.tournament_group_size)
            group_winner = max(group, key=lambda s: s.fitness)
            mating_pool.append(group_winner)
        return mating_pool

    def selection(self) -> list[Specimen]:
        """
        Perform given selection type to get new mating pool.
        """
        m = self.params.selection_mode

        if m == 'rank':
            return self.rank_selection()
        elif m == 'roulette':
            return self.roulette_selection()
        elif m == 'tournament':
            return self.tournament_selection()
        
        raise NotImplementedError("Unknown selection mode")
    
    def mutate_group(self, to_be_mutated: list[Specimen]) -> None:
        """
        In-place mutation of given specimens according to mutation probability params.
        """
        for specimen in to_be_mutated:
            if random() < self.params.specimen_mutation_probability:
                specimen.mutate(
                    self.params.trait_mutation_percentage)
                
    
                
    
        
    