from __future__ import annotations
from utils import FromJSON, join_dataset_to_str
import random
from typing import Literal, Optional
from prompt import Prompt, PromptParams, Trait
import prompt_seed_phrases as seed
import selection_mechanisms as sm
from stats import stats
from time import time
import os
import json
class EvoParams(FromJSON): 
    default_path = "conf/_evo/{}.json"
    """
    Collection of parameters for evolutionary algorithm
    """
    def __init__(self,
                 initial_population_size: int,
                 mating_pool_size: int,
                 max_iters: int,
                 selection_mode: Literal['rank', 'roulette', 'tournament', 'exp_rank', 'random'],
                 similarity_th: float,
                 train_batch_size: int,
                 name: str)-> None:
        self.initial_population_size = initial_population_size
        self.mating_pool_size = mating_pool_size
        assert self.initial_population_size >= self.mating_pool_size
        self.max_iters = max_iters
        self.selection_mode = selection_mode
        self.train_batch_size = train_batch_size
        self.similarity_th=similarity_th
        self.prompt_params: Optional[PromptParams]=None
        self.name = name


class EvolutionaryAlgorithm():
    """
    Class for prompt optimalization using evolutionary algorithms.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.population: list[Prompt] = []
        self.random_baseline: list[Prompt] = []
        self.population_through_steps: list[list[Prompt]] = []
        self.params = config.evoparams
        self.target_pop_size = self.params.initial_population_size
        self.metaprompts = config.metaprompt_set
        self.trait_ids = list(self.metaprompts.keys())
        self.gen = config.model_api.generate
        self.tasks = config.task_toolkit.splits.train
        self.step = 0

    def run(self) -> None:
        """
        Run EvolutionaryAlgorithm to max iterations.
        """
        if self.config.continue_run:
            self.load_population(self.config.ident)
        else:
            self.populate()

        for i in range(self.params.max_iters):
            stats.start_step()
            t = time()
            self.ga_step()
            stats.add_to_current_step({
                "Step duration": time() - t
            })
            print(f"Step {self.step} (this run: {i}) complete")
            self.step += 1
    
    def populate(self) -> None:
        """
        Generate initial prompts based on task input/output samples.
        Creates many random examples to create a strong baseline for evaluation of the evolution progress.
        """
        for _ in range(self.params.initial_population_size * (self.params.max_iters+1)):
            new = self.create_prompt_lamarck()
            self.random_baseline.append(new)
        self.population = self.random_baseline[-self.params.initial_population_size:]
        self.random_baseline = self.random_baseline[:-self.params.initial_population_size]
        for p in self.random_baseline:
            p.log("baseline")
            
    def load_population(self, ident) -> None:
        steps_path = f"runs/{ident}/steps"
        files = list(sorted(os.listdir(steps_path)))

        # load baselines and add new ones to keep computation equivalent
        with open(steps_path+"/baseline.ndjson", "r") as f:
            self.random_baseline = [Prompt(json.loads(l), self.params.prompt_params)
                                    for l in f.readlines()]
        for _ in range(self.params.initial_population_size * self.params.max_iters):
            self.random_baseline.append(self.create_prompt_lamarck())

        added = False
        for fn in files[1:]:
            with open(steps_path+f"/{fn}", "r") as f:
                this_step = [Prompt(json.loads(l), self.params.prompt_params) for l in f.readlines()]
                print("loaded prompts from", fn)
                if len(this_step) == self.params.initial_population_size:
                    added = True
                    self.population_through_steps.append(this_step)
                    print("adding to pop steps")
        if added and len(self.population_through_steps) > 0: 
            self.step = len(self.population_through_steps)
            print("starting from", self.step)
            self.population = [p.copy(new_id=False) for p in self.population_through_steps[-1]]
        else:
            raise IndexError(f"Error when loading pop through steps {added=}, {len(self.population_through_steps)=}")

    def repopulate(self) -> None:
        new_prompts = []
        for _ in range(self.params.initial_population_size - self.pop_size):
            new_prompts.append(self.create_prompt_lamarck())
        self.population += new_prompts

    
    def create_prompt_lamarck(self) -> Prompt:
        """
        Construct a new prompt specimen using metainstructions with task in/out examples.
        """
        def examples_prep_helper():
            examples = self.config.task_toolkit.splits.infer.shuffle()
            return join_dataset_to_str(examples)

        traits = []
        for trait_metaprompt in self.metaprompts.trait_metaprompts:
            trait_text = self.gen(
                trait_metaprompt.format({
                "metapersona": self.metapersona,
                "examples": examples_prep_helper(),
                "metastyle": self.metastyle,
                "length": seed.random_length(self.metaprompts.settings["points_range"],
                                             self.metaprompts.settings["sentences_per_point_range"])
                })
            )

            traits.append(
                Trait(trait_text)
            )

        res = Prompt(traits, self.params.prompt_params)
        stats.add_to_current_step({"New prompt generation": 1})
        return res
    
    def ga_step(self) -> None:
        """
        One step of Genetic Algorithm.
        """
        self.deduplicate()
        self.score_prompts()
        self.population_through_steps.append([p.copy(new_id=False) for p in self.population])
        print(f"just added a step with {len(self.population_through_steps[-1])} prompts")
        self.mating_pool = self.selection()
        
        # GA crossover procedure
        lotto = range(self.params.mating_pool_size)
        offsprings = []
        while len(offsprings) < self.target_pop_size:
            i1, i2 = random.sample(lotto, 2)
            s1, s2 = self.mating_pool[i1], self.mating_pool[i2]

            res = self.crossover(s1, s2)

            if "mutate" in self.metaprompts.keys():
                # Crossover and mutation happen separately
                self.mutate(res)

            offsprings.append(res)

        self.population = offsprings

    def score_prompts(self) -> None:
        task_batch_ix = random.sample(range(len(self.tasks)), self.params.train_batch_size)
        task_batch = self.tasks.select(task_batch_ix)
        
        for s in self.population:
            s.calculate_fitness(task_batch)
            s.log(self.step)

    def deduplicate(self) -> None:
        # apply bert deduplication before fitness based selection
        self.population = sm.filter_similar(self.population, self.params)
        self.repopulate()

    def selection(self) -> list[Prompt]:
        """
        Perform given selection type to get new mating pool.
        """        

        selection_function = sm.sm_dict[self.params.selection_mode]
        return selection_function(self.population, self.params)
    

    def mutate(self, prompt: Prompt) -> None:
        """
        In-place mutation of prompt trait-by-trait.
        """
        
        for ix, trait in enumerate(prompt.traits):
            if trait.evolved:
                prompt.traits[ix].text = self.gen(
                    self.metaprompts.get("mutation").format({
                        "sequence": trait,
                        "metastyle": self.metastyle
                    })
                )

    def crossover(self, prompt1: Prompt, prompt2: Prompt) -> Prompt:
        """
        Create an offspring by combining this prompt's traits with other prompt
        """
        assert prompt1.n_traits == prompt2.n_traits
        
        id1 = prompt1.id
        res = prompt1.copy()
        for ix in range(res.n_traits):
            if prompt1.traits[ix].evolved and prompt2.traits[ix].evolved:
                t1, t2 = res.traits[ix], prompt2.traits[ix]

                res.traits[ix].text = self.gen(
                    self.metaprompts.get("crossover").format({
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
        return random.choice(personas) if self.config.metaprompt_set.settings["metapersonas"] else ""
    
    @property
    def metastyle(self) -> str:    
        return random.choice(seed.wording_styles) if self.config.metaprompt_set.settings["metastyles"] else ""

    @property
    def pop_size(self) -> int:
        return len(self.population)

    @property
    def all_prompts(self) -> list[Prompt]:
        """
        Flatten 2d population through steps into single list.
        """
        return [prompt for step_pop in self.population_through_steps for prompt in step_pop]