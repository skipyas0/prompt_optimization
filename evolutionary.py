from __future__ import annotations
import random
from typing import Literal, Callable, Optional
from prompt import Prompt, PromptParams, Trait
from datasets import Dataset
import prompt_seed_phrases as seed
import metaprompt as mp

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
                 evolution_mode: Literal['GA']='GA',
                 selection_mode: Literal['rank', 'roulette', 'tournament']='rank',
                 tournament_group_size: int=3,
                 train_batch_size: int=5,
                 log: bool = True,
                 combine_co_mut: bool = True,
                 scorer: str = "ask_llm_to_compare",
                 filter_similar_method: str = "None",
                 filter_th: float = 0.95,
                 examples_for_initial_generation: str = "",
                 repop_method_proportion: float = 1.0,
                 metapersonas: bool = False,
                 metastyles: bool = False)-> None:
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
        self.bert = None
        self.similarity_scorer = self.get_similarity_scoring_handle()
        self.repop_method_proportion=repop_method_proportion
        self.prompt_params: Optional[PromptParams]=None
        self.metapersonas = metapersonas
        self.metastyles = metastyles
        

    def get_similarity_scoring_handle(self) -> Optional[Callable[[Prompt, Prompt], float]]:
        """
        Instantiate a handle for the text comparison method specified in params.
        """
        fsm = self.filter_similar_method
        if fsm == 'bert':
            from bert import bert
            self.bert = bert
            return lambda a, b: self.bert.cos_sim_precalc(a.bert_embedding, b.bert_embedding)
        if fsm == 'rouge':
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            return lambda a,b: scorer.score(str(a),str(b))["rougeL"].fmeasure
        if fsm == 'levenshtein':
            from Levenshtein import ratio
            return lambda a,b: ratio(str(a), str(b))
        return None

import selection_mechanisms as sm

class EvolutionaryAlgorithm():
    """
    Class for prompt optimalization using evolutionary algorithms.
    """

    def __init__(self, params: EvoParams, generation_handle: Callable[[str], str], tasks: Dataset) -> None:
        assert params.prompt_params, "Initialize EvoParams.prompt_params with PromptParams object before use."
        self.population: list[Prompt] = []
        self.population_through_steps: list[list[Prompt]] = []
        self.params = params
        self.target_pop_size = self.params.initial_population_size
        self.metaprompts = mp.metaprompts_dict
        self.gen = generation_handle 
        self.tasks = tasks
        self.step = self.ga_step if self.params.evolution_mode=='GA' else None

    def run(self) -> None:
        """
        Run EvolutionaryAlgorithm to max iterations.
        """
        for i in range(self.params.max_iters):
            self.step()
            self.population_through_steps.append(self.population)

    def populate(self) -> None:
        """
        Generate initial prompts based on task input/output samples.
        """
        for _ in range(self.params.initial_population_size):
            new = self.create_prompt_lamarck()    
            self.population.append(new)

    def repopulate(self) -> None:
        #print(f"in repopulate, will add {self.params.mating_pool_size - self.pop_size}")
        new_prompts = []
        for _ in range(self.params.mating_pool_size - self.pop_size):
            if self.pop_size < 1 or random.random() < self.params.repop_method_proportion:
                new = self.create_prompt_lamarck()
            else:
                template = random.choice(self.population)
                new = self.create_prompt_mutate_from_template(template)
            new_prompts.append(new)
        self.population += new_prompts

    def create_prompt_lamarck(self) -> Prompt:
        """
        Construct a new prompt specimen using metainstructions with task in/out examples.
        """

        traits = []
        for trait_name in self.params.trait_ids:
            trait_text = self.gen(
                self.metaprompts[trait_name].format({
                "metapersona": self.metapersona,
                "examples": self.params.examples_for_initial_generation,
                "metastyle": self.metastyle
                })
            )

            traits.append(
                Trait(trait_text)
            )

        res = Prompt(traits, self.params.prompt_params)
        if self.params.filter_similar_method == 'bert':
            res.bert_embedding = self.params.bert.get_bert_embedding(str(res))
        return res
    
    def create_prompt_mutate_from_template(self, template: Prompt) -> Prompt:
        """
        Construct a new prompt by applying the mutation operator on a given template prompt.
        """

        res = template.copy()
        self.mutate(res)
        res.parent_ids = [template.id]
        if self.params.filter_similar_method == 'bert':
            res.bert_embedding = self.params.bert.get_bert_embedding(str(res))
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
            i1, i2 = random.sample(lotto, 2)
            s1, s2 = self.population[i1], self.population[i2]

            res = self.crossover(s1, s2)

            if not self.params.combine_co_mut:
                # Crossover and mutation happen separately
                if random.random() < self.params.prompt_mutation_probability:
                        self.mutate(res)

            if self.params.filter_similar_method == 'bert':
                res.bert_embedding = self.params.bert.get_bert_embedding(str(res))
            offsprings.append(res)

        self.population += offsprings

        self.target_pop_size = max(self.params.mating_pool_size,
                            self.target_pop_size + self.params.population_change_rate)

   
    def selection(self) -> list[Prompt]:
        """
        Perform given selection type to get new mating pool.
        """
        task_batch_ix = random.sample(range(len(self.tasks)), self.params.train_batch_size)
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
            if random.random() < self.params.prompt_mutation_probability:
                self.mutate(prompt)
    
    def mutate(self, prompt: Prompt) -> None:
        """
        In-place mutation of prompt trait-by-trait.
        """
        
        for ix, trait in enumerate(prompt.traits):
            if trait.evolved:
                prompt.traits[ix].text = self.gen(
                    self.metaprompts['mutation'].format({
                        "sequence": trait,
                        "metastyle": self.metastyle
                    })
                )

    def crossover(self, prompt1: Prompt, prompt2: Prompt) -> Prompt:
        """
        Create an offspring by combining this prompt's traits with other prompt
        """
        assert prompt1.n_traits == prompt2.n_traits
        template_name = "mutated_crossover" if self.params.combine_co_mut else "crossover"
        
        id1 = prompt1.id
        res = prompt1.copy()
        for ix in range(res.n_traits):
            if prompt1.traits[ix].evolved and prompt2.traits[ix].evolved:
                t1, t2 = res.traits[ix], prompt2.traits[ix]

                res.traits[ix].text = self.gen(
                    self.metaprompts[template_name].format({
                        "sequence1": t1,
                        "sequence2": t2,
                        "metastyle": self.metastyle
                    })
                )

        res.parent_ids = [id1, prompt2.id]
        res.generation_number += 1
        return res
    
    @property
    def metapersona(self) -> str:
        personas = seed.manager_personas if random.random() < 0.5 else seed.solver_personas
        return random.choice(personas) if self.params.metapersonas else ""
    
    @property
    def metastyle(self) -> str:    
        return random.choice(seed.wording_styles) if self.params.metastyles else ""

    @property
    def pop_size(self) -> int:
        return len(self.population)

    @property
    def all_prompts(self) -> list[Prompt]:
        """
        Flatten 2d population through steps into single list.
        """
        return [prompt for step_pop in self.population_through_steps for prompt in step_pop]