from __future__ import annotations
import re
import json
from datasets import Dataset, load_dataset, Value
from fitness_functions import ff_dict
from utils import FromJSON
from metaprompt import MetapromptSet
from collections import namedtuple
from typing import Callable, Any
class TaskToolkit(FromJSON):
    default_path = "conf/_task/{}.json"
    def __init__(self, 
                 dataset: str, 
                 split_lengths: tuple[int, int,int], 
                 metaprompt_set: str, 
                 ans_type: str,
                 formatting_suffix: str, 
                 scorer: str,
                 name:str) -> None:
        """ 
        Class containing all info about the task the EA is run on.
        Separable from evolution parameters.
        """
        self.dataset_name = dataset
        self.split_lengths = split_lengths
        self.metaprompt_set_name = metaprompt_set
        self.ans_type = ans_type
        self.formatting_suffix = formatting_suffix
        self.scorer = scorer
        self.metaprompt_set = MetapromptSet.from_json(metaprompt_set)
        self.name = name

        if self.dataset_name == 'openai/gsm8k':
            self.ds, self.ans_type = load_gsm8k()
        elif re.fullmatch('^maveriq/bigbenchhard/[a-z]+(_[a-z]+)*$', self.dataset_name):
            subset = self.dataset_name.split('/')[-1]
            self.ds, self.ans_type = load_bigbenchhard(subset)
        elif re.fullmatch('^cais/mmlu/[a-z]+(_[a-z]+)*$', self.dataset_name):
            subset = self.dataset_name.split('/')[-1]
            self.ds, self.ans_type = load_mmlu(subset)
        elif self.dataset_name == 'deepmind/code_contests':
            self.ds, self.ans_type = load_code_contests()
        elif re.fullmatch('^livebench/language/[a-z]+(_[a-z]+)*$', self.dataset_name):
            subset = self.dataset_name.split('/')[-1]
            self.ds, self.ans_type = load_livebench_language(subset)
        else:
            raise KeyError(f"Unsupported dataset {self.dataset_name}")
        self.splits = create_splits(self.ds, self.split_lengths)
        self.scoring_function: Callable[[Any, Any], float]  = ff_dict[self.scorer]


def create_splits(dataset: Dataset, split: tuple[int, int, int]) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load dataset and split it according to sample numbers in input tuple.
    """
    a,b,c = split
    total = sum(split)
    if total > len(dataset):
        b = len(dataset) - a - c
        total = a + b + c

    ds = dataset.shuffle(seed=42).select(range(total))
    infer = ds.select(range(a))
    train = ds.select(range(a, a+b))
    test = ds.select(range(a+b, total))
    splits = namedtuple("Splits", ["infer", "train", "test"])
    return splits(infer, train, test)


#### | ################################## | ####
#### | Dataset specific loading functions | ####
#### V ################################## V ####

def load_bigbenchhard(subset):
    ds = load_dataset('maveriq/bigbenchhard', subset, split='train')
    ds = ds.map(map_bigbenchhard, load_from_cache_file=False)
    if subset in ['causal_judgement', 'navigate', 'web_of_lies', 'sports_understanding']:
        ans_type = 'yes-no'
    elif subset in ['snarks', 'disambiguation_qa', 'geometric_shapes', 'hyperbaton', 'movie_recommendation', 'penguins_in_a_table']:
        ans_type = 'choice'
    else:
        raise KeyError(f"Unsupported bigbenchhard subset {subset}")
    def map_bigbenchhard(example):
        if type(example['target']) == list:
            example['target'] = example['target'][0]
        if type(example['input']) == list:    
            example['input'] = example['input'][0]
    
        if re.fullmatch('^[(][A-Z][)]$',example['target']):
            example['target'] = example['target'][1]
        elif example['target'] == 'no':
            example['target'] = 'No'
        elif example['target'] == 'yes':
            example['target'] = 'Yes'
        return {'question': example['input'], 'answer': example['target']} 
    return ds, ans_type

def load_mmlu(subset):
    ds = load_dataset('cais/mmlu', subset, split='test')
    ds = ds.cast_column('answer', Value('string'))
    ds = ds.map(map_mmlu, remove_columns=ds.column_names, load_from_cache_file=False)
    ans_type = 'choice'
    def map_mmlu(example):
        example['question'] = example['question'] + '\n' + '\n'.join([f'{op}: {op_text}' for op, op_text in zip("ABCD", example['choices'])])
        example['answer'] = "ABCD"[int(example['answer'])]
        return {'question': example['question'], 'answer': example['answer']}
    return ds, ans_type

def load_gsm8k():
    ds = load_dataset('openai/gsm8k', 'main', split='train')
    ds = ds.map(map_gsm8k, remove_columns=ds.column_names, load_from_cache_file=False)
    ans_type = 'numeric'

    def map_gsm8k(example):
        example['question'] = example['question']
        example['answer'] = example['answer'].split('####')[1].replace('\xa0', '').strip()
        return {'question': example['question'], 'answer': example['answer']}
    return ds, ans_type

def load_code_contests():
    ds = load_dataset('deepmind/code_contests', split='train')
    ds = ds.filter(lambda ex: ex['difficulty']  == 7 and '<image>' not in ex['description']) # filter easy samples 
    ds = ds.map(map_code_contests, remove_columns=ds.column_names, load_from_cache_file=False)
    ans_type = 'code'
    def map_code_contests(example):
        question = example['description']
        test_inputs = [x.strip().split('\n') for x in example['private_tests']['input']]
        test_outputs = [x.strip() for x in example['private_tests']['output']]
        
        return {
            'question': question,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
        }
    return ds, ans_type

def load_livebench_language(subset):
    def map_livebench_language(example):
        question = example["turns"][0].split('\n')[-1]
        answer = example["ground_truth"]
        return {
            'question': question,
            'answer': answer
        }
    ds = load_dataset('livebench/language', split='test')
    ds = ds.filter(lambda x: x["task"] == subset)
    ds = ds.map(map_livebench_language, remove_columns=ds.column_names, load_from_cache_file=False)
    ans_type = 'text'

    return ds, ans_type