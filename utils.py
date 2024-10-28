from datasets import load_dataset, Dataset, Value
import re
import json
from vllm_api import OpenAIPredictor
from typing import Callable, Literal, Optional
import random 
from bert import bert
from metaprompt import formatting_enforcement_suffixes
def load_log_dict(path: str) -> list[dict]:
    """
    Open .ndjson and load it as list of dicts
    """
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))  # Convert each line to a dictionary
    return data

def load_dataset_and_preprocess(ds_name: str) -> tuple[str, Dataset]:
    """
    Downloads one of the supported datasets and preprocesses it.
    """
    if ds_name == 'openai/gsm8k':
        ds = load_dataset(ds_name, 'main', split='train')
        ans_type = 'numeric'
    elif re.fullmatch('^maveriq/bigbenchhard/[a-z]+(_[a-z]+)*$', ds_name):
        subset = ds_name.split('/')[-1]
        ds = load_dataset('maveriq/bigbenchhard', subset, split='train')
        ds = ds.map(map_bigbenchhard)
        if subset in ['causal_judgement', 'navigate', 'web_of_lies', 'sports_understanding']:
            ans_type = 'yes-no'
        elif subset in ['snarks', 'disambiguation_qa', 'geometric_shapes', 'hyperbaton', 'movie_recommendation', 'penguins_in_a_table']:
            ans_type = 'choice'
        else:
            raise KeyError(f"Unsupported bigbenchhard subset {subset}")
    elif ds_name == 'GBaker/MedQA-USMLE-4-options':
        ds = load_dataset(ds_name, 'default', split='train')
        ds = ds.map(map_medqa_usmle, remove_columns=ds.column_names)
        ans_type = 'choice'
    elif re.fullmatch('^cais/mmlu/[a-z]+(_[a-z]+)*$', ds_name):
        subset = ds_name.split('/')[-1]
        ds = load_dataset('cais/mmlu', subset, split='test')
        ds = ds.cast_column('answer', Value('string'))
        ds = ds.map(map_mmlu, remove_columns=ds.column_names)
        ans_type = 'choice'
    else:
        raise KeyError(f"Unsupported dataset {ds_name}")
    return ans_type, ds

def load_splits(ds_name: str, split: tuple[int, int, int]) -> tuple[str, Dataset, Dataset, Dataset]:
    """
    Load dataset and split it according to sample numbers in input tuple.
    """
    ans_type, ds = load_dataset_and_preprocess(ds_name)
    suff = formatting_enforcement_suffixes[ans_type]
    ds = ds.shuffle(seed=42).select(range(sum(split)))
    infer = ds.select(range(split[0]))
    train = ds.select(range(split[0], sum(split[:2])))
    test = ds.select(range(sum(split[:2]), sum(split)))
    return suff, infer, train, test

def join_dataset_to_str(dataset: Dataset, insertion_token: str, insertion_position: Literal["prefix", "suffix"] = "prefix") -> str:   
    """
    Join samples from datasets to single string with <in> <out> html-like tags.
    """

    res = ""

    features = list(dataset.features.keys())
    q, a = features[0], features[1]

    for i, sample in enumerate(dataset):
        res += f"<{i}>\n"
        res += insertion_token if insertion_position == "prefix" else "Follow these steps and solve the problem in a logical fashion.\n 1. Analyze the problem.\n 2. Create a plan to solve it.\n 3. Follow the plan and explain your steps.\n 4. Give your final answer.\n"
        res += f"<{q}> {sample[q]} </{q}>\n"
        res += insertion_token if insertion_position == "suffix" else "Let's think step by step.\n"
        res += f"<{a}> {sample[a]} </{a}>\n"
        res += f"</{i}>\n"

    return res

def parse_verdict(text: str) -> str:
    """
    Parse [[[verdict]]]
    """
    pattern = r'\[\[\[([a-z]+ ?[a-z]*)\]\]\]'# <- only lowercase letters 
    #any char - r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches

def parse_answer(text: str) -> str:
    """
    Parse [[answer]]
    """
    pattern = r'\[\[(.*)\]\]'
    #any char - r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches

def log_usage(log_file: str, input: str | tuple[str, str], output: str | float) -> None:
    """
    Add entry about handle usage to log file.
    """
    if type(output) == str:
        log_entry = {
            'type': "usage",
            'in': input,
            'out': output,
        }
    else:
        log_entry = {
            'type': "score",
            'ground': input[0],
            'in': input[1],
            'out': output,
        }

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def create_api_handles(api: Optional[OpenAIPredictor], log_file: str, scorer: str) -> tuple[Callable[[str, float], str], Callable[[str], float]]:
    if api is None:
        #fast debug to replace LLM calls 
        def scramble(input: str, _: float) -> str:
            char_list = list(input)
            random.shuffle(char_list)
            return ''.join(char_list)
        usage_helper = scramble
        score_helper = lambda _0, _1: random.random()
    else:
        import fitness_functions as ff
        usage_helper = lambda prompt, temp: api.predict(question=prompt, temp=temp)
        if scorer == "ask_llm_to_compare":
            score_helper = lambda ground, x: ff.ask_llm_to_compare(ground, x, usage_handle)
        elif scorer == "binary_match":
            score_helper = lambda ground, x: ff.binary_match(ground, x)
        elif scorer == "levenshtein":
            from Levenshtein import ratio            
            score_helper = ratio
        elif scorer == "rouge":
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            score_helper = lambda a,b: scorer.score(a,b)["rougeL"].fmeasure 
        elif scorer == "bert":
            score_helper = bert.bert_cosine_similarity

    def usage_handle(input: str, temp: float) -> str:
        out = usage_helper(input, temp)
        log_usage(log_file, input, out)
        return out
    
    def score_handle(ground: str, input: str) -> float:

        # if this is the gsm8k dataset, the answer is EXPLANATION #### NUMERIC ANSWER
        spl = ground.split("####")
        if len(spl) == 2:
            ground = spl[1].strip()

        out = score_helper(ground, input)
        log_usage(log_file, (ground, input), out)
        return out
    
    return usage_handle, score_handle


def find_key_by_value(d, x):
    for key, value in d.items():
        if value == x:
            return key
    return None  

def map_medqa_usmle(example):
    options_text = "\n".join(f'{k}: {v}' for k,v in example['options'].items()) 
    example['question'] = example['question'] + "\nOptions:\n" + options_text
    example['answer'] = find_key_by_value(example['options'], example['answer'])
    return {'question': example['question'], 'answer': example['answer']}  # Only keep these two fields

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

def map_mmlu(example):
    example['question'] = example['question'] + '\n' + '\n'.join([f'{op}: {op_text}' for op, op_text in zip("ABCD", example['choices'])])
    example['answer'] = "ABCD"[int(example['answer'])]
    return {'question': example['question'], 'answer': example['answer']}
