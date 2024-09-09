from __future__ import annotations
from random import shuffle
from math import ceil
from typing import TypeVar, Generic

T = TypeVar('T')
class Specimen():
    """
    Bundle of traits to be evaluated and optimized via evolution in a evolutionary algorithm.
    Template class - ToBeImplemented:
        Specimen._trait_mutate (GA and DE)
        Specimen._trait_crossover (GA and DE)
        Specimen._trait_difference (DE only)
        Specimen._trait_addition  (DE only)
        Specimen._fitness (GA and DE)
        Specimen._generate_traits (GA and DE)
    """
    def __init__(self, traits: list[Generic[T]]) -> None:
        self.traits = traits
        self.n_traits = len(self.traits)
        self.fitness = float('-inf')
        self.generation_number = 1

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
        indices_to_mutate = list(range(self.n_traits))
        shuffle(indices_to_mutate)
        indices_to_mutate = indices_to_mutate[:n_to_mutate]
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
        new = type(other)(self.traits.copy())
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
        new = type(other1)(self.traits.copy())
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
