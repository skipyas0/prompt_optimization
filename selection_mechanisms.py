from prompt import Prompt
from evolutionary import EvoParams
import torch
from random import sample


def rank_selection(population: list[Prompt], params: EvoParams) -> list[Prompt]:
        """
        Individuals are ranked based on their fitness, and selection is done based on rank.
        """
        mating_pool = sorted(population, 
                     key=lambda s: s.fitness, 
                     reverse=True)[:params.mating_pool_size]
        
        return mating_pool
        
    
def roulette_selection(population: list[Prompt], params: EvoParams) -> list[Prompt]:
    """
    Individuals are selected with a probability proportional to their fitness. 
    The better the fitness, the higher the probability of being selected.
    """

    sm_scores = torch.softmax(torch.tensor([s.fitness for s in population]), dim=0)
    counts = map(int,(10 * sm_scores / sm_scores.min()).ceil().tolist())

    mating_pool = sample(population, params.mating_pool_size,
                            counts=counts)
    
    return mating_pool

def tournament_selection(population: list[Prompt], params: EvoParams) -> list[Prompt]:
    """
    A group of individuals (a tournament) is randomly chosen from the population. 
    The best individual in this group is selected. 
    This is repeated until the desired number of individuals is chosen.
    """
    mating_pool = []
    for _ in range(params.mating_pool_size):
        group = sample(population, params.tournament_group_size)
        group_winner = max(group, key=lambda s: s.fitness)
        mating_pool.append(group_winner)
    return mating_pool

def random_selection(population: list[Prompt], params: EvoParams) -> list[Prompt]:
    """
    Select individuals randomly, unrelated to their fitness.
    """
    mating_pool = sample(population, params.mating_pool_size)
    
    return mating_pool

def filter_similar(population: list[Prompt], params: EvoParams) -> list[Prompt]:
    """
    Deduplicate population based on a given similarity metric eg.: bert cosine similarity.
    """
    ix_to_pop = set()
    print(f"Before {len(population)=}")
    # mark prompts that have a duplicate for removal
    for i, p1 in enumerate(population):
        for p2 in population[i+1:]:
             sim = params.similarity_scorer(str(p1), str(p2))
             print(f"{str(p1)[:15]=} and {str(p2)[:15]=} have similarity {sim} -> {'popping' if sim > params.filter_th else 'keeping'}")
             if sim > params.filter_th:
                  ix_to_pop.add(i)
                  break
             
    # pop marked prompts
    for ix in sorted(list(ix_to_pop), reverse=True):
         population.pop(ix)
    print(f"After {len(population)=}")
    return population