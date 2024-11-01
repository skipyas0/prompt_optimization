from __future__ import annotations
from typing import Callable, Literal
from random import randint
from datasets import Dataset
import json
import metaprompt
from stats import stats

universal_suffix = """After your explanation, make sure you put your final answer in two pairs of square brackets. 
<example>
...
And for the above reasons, the solution is ANSWER.
[[ANSWER]]
</example>"""


class Trait:
    def __init__(self, text: str = "", position: Literal['prefix', 'suffix'] = 'prefix', evolved: bool = True) -> None:
        self.text = text
        self.position = position
        self.evolved = evolved

    def __str__(self) -> str:
        return self.text

    def copy(self) -> Trait:
        return Trait(
            self.text,
            self.position,
            self.evolved
        )

class PromptParams():
    def __init__(self, usage_handle: Callable[[str], str], evalutation_handle: Callable[[str, str], float], log_file: str, format_enforcement_suffix: str) -> None:
        self.usage_handle = usage_handle
        self.evaluation_handle = evalutation_handle
        self.log_file = log_file
        self.metaprompt = metaprompt.solve
        self.format_enforment_suffix = format_enforcement_suffix
class Prompt():
    def __init__(self, traits: list[Trait] | dict, params: PromptParams) -> None: 
        if isinstance(traits, dict):   
            self.params = params
            self.traits = traits['traits']
            self.n_traits = len(self.traits)
            self.fitness = traits['avg_fitness']
            self.best_fitness = traits['best_fitness']
            self.generation_number = traits['generation']
            self.result = traits['best_task_result']
            self.id = traits['id']
            self.parent_ids = traits['parent_ids']
        else:
            self.params = params
            self.traits = traits
            self.n_traits = len(self.traits)
            self.fitness = float('-inf')
            self.best_fitness = float('-inf')
            self.generation_number = 1
            self.result = ["",""]
            self.id = randint(10000000, 99999999)
            self.parent_ids = []
        self.bert_embedding = None
        self.maximum_similarity = 0
        self.average_similarity = 0

    def prefix_part(self) -> str:
        return '\n'.join(map(str, filter(lambda x: x.position == 'prefix', self.traits)))
    
    def suffix_part(self) -> str:
        return '\n'.join(map(str, filter(lambda x: x.position == 'suffix', self.traits)))

    def format(self, task: str) -> str:
        """
        Insert task in the prompt template and output the complete prompt
        """
        return self.prefix_part() + '\n<task>\n' + task +'\n</task>\n' + self.suffix_part() 
    
    def __str__(self) -> str:
        """
        Join prompt traits into a formattale string
        """
        return self.prefix_part() + '\n<task>\n{}\n</task>\n' + self.suffix_part()

    def calculate_fitness(self, batch: Dataset) -> float:
        """
        Evalute mean performance over batch of tasks
        """
        results = [self.params.usage_handle(self.params.metaprompt.format({
                "preamble": "", 
                "prefix_instructions": self.prefix_part(), 
                "task": task["question"], 
                "suffix_instructions": self.suffix_part(), 
                "universal_suffix": self.params.format_enforment_suffix
            })) 
            for task in batch]
        
        ground_truths = [task['answer'] for task in batch]
        fitness_scores = [self.params.evaluation_handle(ground, gen) for ground, gen in zip(ground_truths, results)]
        self.best_fitness, self.result = max(zip(fitness_scores, results)) # save result with best performance on 
        self.fitness = sum(fitness_scores) / len(fitness_scores)
        stats.append_to_current_step({"Fitness in training": self.fitness})
        return self.fitness

    def copy(self) -> Prompt:
        new = Prompt([trait.copy() for trait in self.traits], self.params)
        new.generation_number = self.generation_number
        new.fitness = self.fitness
        new.best_fitness = self.best_fitness
        new.id = self.id
        new.parent_ids = self.parent_ids
        return new
    
        
    def log(self) -> None:
        """
        Add entry about self to .ndjson defined in PromptParams.
        """
        log_entry = {
            'type': "prompt",
            'id': self.id,
            'parent_ids': self.parent_ids,
            'generation': self.generation_number,
            'traits': [(x.text, x.position, x.evolved) for x in self.traits],
            'avg_fitness': self.fitness,
            'best_task_result': self.result,
            'best_fitness': self.best_fitness,
            'average_similarity': float(self.average_similarity),
            'maximum_similarity': float(self.maximum_similarity)
            #'bert_embedding': self.bert_embedding
        }
        with open(self.params.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            #f.write(f"Prompt gen: {self.generation_number}\n\n***\n{str(self)}\n***\nTask with best result:\n{self.result[0]}\nResult:\n{self.result[1]}\n\nScore: {self.fitness}\n\n######")