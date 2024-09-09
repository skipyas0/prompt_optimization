from __future__ import annotations
from random import shuffle, random, sample
from math import ceil
from typing import Literal, Generic
from specimen import Specimen, T

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


class EvolutionaryAlgorithm():
    """
    Universal framework class for evolutionary algorithms for optimization.
    """
    def __init__(self, params: EvoParams) -> None:
        self.population: list[Specimen] = []
        self.params = params
        self.pop_size = self.params.initial_population_size
        

    def populate(self, traits: list[list[Generic[T]]] | None=None) -> None:
        """
        Generate initial population of specimens.
        """
        raise NotImplementedError
    
    def run(self) -> None:
        """
        Run EvolutionaryAlgorithm to max iterations.
        """
        for i in range(self.params.max_iters):
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

            res = s1.crossover(s2)
            res.generation += 1
            offsprings.append(res)

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

            res = basic.crossover(s1)
            res.generation += 1
            offsprings.append(res)

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
                