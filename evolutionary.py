from __future__ import annotations
from random import shuffle, random, sample
from typing import Literal, Callable, Optional
from prompt import Prompt, PromptParams
from datasets import Dataset
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
                 trait_ids: str = ['instructions'],
                 task_insert_ix: int = 1,
                 prompt_mutation_probability: float=1.0,
                 trait_mutation_percentage: float=1.0,
                 max_iters: int=100,
                 evolution_mode: Literal['GA','DE']='GA',
                 selection_mode: Literal['rank', 'roulette', 'tournament']='rank',
                 tournament_group_size: int=3,
                 train_batch_size: int=5,
                 log: bool = True,
                 combine_co_mut: bool = True,
                 scorer: str = "ask_llm_to_compare",
                 filter_similar_method: str = "None",
                 filter_th: float = 0.95,
                 examples_for_initial_generation: str = "",
                 repop_method_proportion: float = 1.0)-> None:
        self.initial_population_size = initial_population_size
        self.population_change_rate = population_change_rate
        self.mating_pool_size = mating_pool_size
        assert self.initial_population_size >= self.mating_pool_size
        self.trait_ids = trait_ids
        self.prompt_trait_n = len(trait_ids)
        self.task_insert_ix = task_insert_ix
        self.prompt_mutation_probability = prompt_mutation_probability
        self.trait_mutation_percentage = trait_mutation_percentage
        self.max_iters = max_iters
        self.evolution_mode = evolution_mode
        self.selection_mode = selection_mode
        self.tournament_group_size = tournament_group_size
        self.train_batch_size = train_batch_size
        self.log=log
        self.combine_co_mut=combine_co_mut
        self.scorer=scorer
        self.filter_similar_method=filter_similar_method
        self.filter_th=filter_th
        self.examples_for_initial_generation = examples_for_initial_generation
        self.similarity_scorer = self.get_similarity_scoring_handle()
        self.repop_method_proportion=repop_method_proportion
        self.prompt_params: Optional[PromptParams]=None

    def get_similarity_scoring_handle(self) -> Optional[Callable[[str, str], float]]:
        """
        Instantiate a handle for the text comparison method specified in params.
        """
        fsm = self.filter_similar_method
        if fsm == 'bert':
            from bert import Bert
            b = Bert()
            return b.bert_cosine_similarity
        if fsm == 'rouge':
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            return lambda a,b: scorer.score(a,b)["rougeL"].fmeasure
        if fsm == 'levenshtein':
            from Levenshtein import ratio
            return ratio
        return None

import selection_mechanisms as sm

class EvolutionaryAlgorithm():
    """
    Class for prompt optimalization using evolutionary algorithms.
    """

    def __init__(self, params: EvoParams, generation_handle: Callable[[str], str], tasks: Dataset) -> None:
        assert params.prompt_params, "Initialize EvoParams.prompt_params with PromptParams object before use."
        self.population: list[Prompt] = []
        self.all_prompts: list[Prompt] = []
        self.params = params
        self.target_pop_size = self.params.initial_population_size
        
        # !!! important !!! handles only accept list of strings
        # necessary for formatting into prompt templates -> asterisk unwrapper
        self.task_specific_handles: dict[str, Callable[[list[str]], str]] = dict() 

        self.gen = generation_handle 
        self.tasks = tasks
        self.load_instructions()
        
        self.step = self.ga_step if self.params.evolution_mode=='GA' else self.de_step

    def run(self) -> None:
        """
        Run EvolutionaryAlgorithm to max iterations.
        """
        for _ in range(self.params.max_iters):
            self.step()
            self.all_prompts += self.population

    def populate(self) -> None:
        """
        Generate initial prompts based on task input/output samples.
        """
        for _ in range(self.params.initial_population_size):
            new = self.create_prompt_lamarck()    
            self.population.append(new)

    def repopulate(self) -> None:
        print(f"in repopulate, will add {self.params.mating_pool_size - self.pop_size}")
        new_prompts = []
        for _ in range(self.params.mating_pool_size - self.pop_size):
            if self.pop_size < 1 or random() < self.params.repop_method_proportion:
                new = self.create_prompt_lamarck()
            else:
                template = sample(self.population, 1)
                new = self.create_prompt_mutate_from_template(template)
            new_prompts.append(new)
        self.population += new_prompts

    def create_prompt_lamarck(self) -> Prompt:
        """
        Construct a new prompt specimen using metainstructions with task in/out examples.
        """
        print("Creating fresh prompt")
        traits = []
        for tr in self.params.trait_ids:
            traits.append(self.task_specific_handles[tr]([self.params.examples_for_initial_generation]))
        
        return Prompt(traits, self.params.prompt_params)
    
    def create_prompt_mutate_from_template(self, template: Prompt) -> Prompt:
        """
        Construct a new prompt by applying the mutation operator on a given template prompt.
        """

        res = template.copy()
        op.mutate(res, self.task_specific_handles["mutation"])
        res.parent_ids = [template.id]
        return res

    def ga_step(self) -> None:
        """
        One step of Genetic Algorithm.
        """
        self.population = self.selection()
        
        # GA crossover procedure
        lotto = range(self.params.mating_pool_size)
        offsprings = []
        while self.pop_size + len(offsprings) < self.target_pop_size:
            i1, i2 = sample(lotto, 2)
            s1, s2 = self.population[i1], self.population[i2]

            if self.params.combine_co_mut:
                res = op.crossover(s1, s2, self.task_specific_handles['mutated_crossover'])
            else:
                res = op.crossover(s1, s2, self.task_specific_handles['crossover'])
                if random() < self.params.prompt_mutation_probability:
                    op.mutate(res, self.task_specific_handles['mutation'])
            
            offsprings.append(res)

        self.population += offsprings

        self.target_pop_size = max(self.params.mating_pool_size,
                            self.target_pop_size + self.params.population_change_rate)

    def de_step(self) -> None:
        """
        One step of Differential Evolution.
        """
        selected_prompts = self.selection()

        handles = tuple(self.task_specific_handles[s] for s in ['de1', 'de2', 'de3'])

        basic = selected_prompts[-1]
        lotto = range(self.params.mating_pool_size - 1)
        offsprings = []
        while len(self.population) + len(offsprings) < self.target_pop_size:
            # New offspring is Crossover(Mutate(s1 - s2) + s3, basic)
            i1, i2, i3 = sample(lotto, 3)
            prompts = (self.population[i1], self.population[i2], self.population[i3], basic)
            res = op.de_combination(prompts, handles)

            offsprings.append(res)

        self.population += offsprings

        # shuffle to make sure a random basic prompt is being chosen
        shuffle(self.population)
        self.target_pop_size = max(self.params.mating_pool_size,
                            self.target_pop_size + self.params.population_change_rate)

    def selection(self) -> list[Prompt]:
        """
        Perform given selection type to get new mating pool.
        """
        task_batch_ix = sample(range(len(self.tasks)), self.params.train_batch_size)
        task_batch = self.tasks.select(task_batch_ix)
        
        for s in self.population:
            s.calculate_fitness(task_batch)
            if self.params.log:
                s.log()

        # apply specified deduplication method before fitness based selection
        if self.params.similarity_scorer:
            self.population = sm.filter_similar(self.population, self.params)
            self.repopulate()

        m = self.params.selection_mode

        if m == 'rank':
            return sm.rank_selection(self.population, self.params)
        elif m == 'roulette':
            return sm.roulette_selection(self.population, self.params)
        else:
            return sm.tournament_selection(self.population, self.params)
    
    def mutate_group(self, to_be_mutated: list[Prompt]) -> None:
        """
        In-place mutation of given prompts according to mutation probability params.
        """
        for prompt in to_be_mutated:
            if random() < self.params.prompt_mutation_probability:
                op.mutate(prompt, self.task_specific_handles['mutation'])
                
    def load_instructions(self) -> None:
        """
        Access prewritten instructions to be used in genetic operators.
        """
        curr_path = os.getcwd()
        path = curr_path + '/metaprompts/'
        for fn in os.listdir(path):
            op_name = fn.split('.')[0]
            with open(path + fn, 'r') as file:
                instructions = file.read()

                # Store a handle to be used with 1 or 2 prompt-traits
                # Unpack them and format them into the instruction template - it has 1 or 2 {} brackets for formatting
                def handle(s: list[str], instructions=instructions) -> str:
                    prompt = instructions.format(*s)
                    return self.gen(prompt)
                 
                self.task_specific_handles[op_name] = handle

        assert len(self.task_specific_handles.keys()) == len(os.listdir(path))

    @property
    def pop_size(self) -> int:
        return len(self.population)