from random import shuffle, random
from math import ceil
from __future__ import annotations
from typing import Callable, TypeVar, Generic

"""
TODO:
Simplify 
Trait and GeneticOperators are unnecessary
Implement in Specimen class

Finish differential evolution
"""




class EvoParams(): 
    """
    Collection of parameters for evolutionary algorithm
    """
    def __init__(self,
                 specimen_mutation_probability=0.2,
                 trait_mutation_percentage=1.0,
                 selection_throwaway_percentage=0.5,
                 max_iters=100) -> None:
        self.specimen_mutation_probability = specimen_mutation_probability
        self.trait_mutation_percentage = trait_mutation_percentage
        self.selection_throwaway_percentage = selection_throwaway_percentage
        self.max_iters = max_iters

T = TypeVar('T')
class Trait():
    """
    Container for generic traits of a specimen in a genetic algorithm to be mutated and combined with other traits.
    """
    def __init__(self, data: Generic[T]) -> None:
        self.type = Generic[T]
        self.data = data



class Specimen():
    """
    Bundle of traits to be evaluated and optimized via evolution in a genetic algorithm.
    """
    def __init__(self, traits: list[Trait]) -> None:
        self.traits = traits
        self.n_traits = len(self.traits)

    def mutate(self, trait_mutation_percentage: float, trait_mutation_function: Callable[[Trait], Trait]):
        n_to_mutate = ceil(trait_mutation_percentage * self.n_traits)
        indices_to_mutate = shuffle(list(range(self.n_traits)))[:n_to_mutate]
        for ix in indices_to_mutate:
            self.traits[ix] = trait_mutation_function(self.traits[ix])

    def crossover(self, other: Specimen, trait_crossover_function: Callable[[Trait, Trait], Trait]) -> Specimen:
        assert self.n_traits == other.n_traits
        for ix, trait in enumerate(self.traits):
            self.traits[ix] = trait_crossover_function(trait, other.traits[ix])
        return Specimen(self.traits)

class GeneticOperators():
    """
    Template class for implementation of genetic operators of a evolutionary algorithm.
    """
    def __init__(self, params: EvoParams) -> None:
        self.params = params

    def mutate(self, trait: Trait) -> Trait:
        raise NotImplementedError
        
    def crossover(self, trait1: Trait, trait2: Trait) -> Trait:
        assert trait1.type == trait2.type
        raise NotImplementedError
    
    def fitness(self, specimen: Specimen) -> float:
        raise NotImplementedError
    
    def selection(self, population: list[float], throwaway: float) -> list[int]:
        raise NotImplementedError

    def difference(self, trait1: Trait, trait2: Trait) -> Trait:
        assert trait1.type == trait2.type
        raise NotImplementedError

class EvolutionaryAlgorithm():
    def __init__(self, 
                 population: list[Specimen], 
                 params: EvoParams, 
                 genetics: GeneticOperators,
                 mode: str = 'GA') -> None:
        assert mode in ['GA', 'DE'] # genetic algorithm or differential evolution
        self.mode = mode
        self.population = population
        self.params = params
        self.genetics = genetics

    def run(self) -> None:
        for _ in range(self.params.max_iters):
            self.step()

    def step(self) -> None:
        if self.mode == 'GA':
            self.ga_step()
        elif self.mode == 'DE':
            self.de_step()

    
    def ga_step(self) -> None:
        """
        One step of Genetic Algorithm
        """
        selected_specimens = self.select_best()

        from_crossover = []
        for i, s1 in enumerate(selected_specimens):
            for s2 in selected_specimens[i:]:
                from_crossover.append(s1.crossover(s2, self.genetics.crossover))
        self.population = self.mutation(selected_specimens + from_crossover)


    def de_step(self) -> None:
        """
        One step of Differential Evolution
        """
        selected_specimens = self.select_best()

        
    def select_best(self) -> list[Specimen]:
        fitness_scores = [self.genetics.fitness(s) for s in self.population]

        stp = self.params.selection_throwaway_percentage
        selected_indices = self.genetics.selection(fitness_scores, stp)
        selected_specimens = [self.population.pop(ix) for ix in sorted(selected_indices, reverse=True)]
        shuffle(selected_specimens)

        return selected_specimens
    
    def mutation(self, to_be_mutated: list[Specimen]) -> list[Specimen]:
        for specimen in to_be_mutated:
            if random() < self.params.specimen_mutation_probability:
                specimen.mutate(
                    self.params.trait_mutation_percentage,
                    self.genetics.mutate)
        return to_be_mutated
                
    
                
    
        
    