from typing import Callable
from utils import parse_verdict
from random import random
def simple_list_intersection(ground: str, sample: str, delimeter: str = 'l') -> float:
    """
    Expects two strings, both delimited by delimeter -> uses them like lists.
    Outputs number of common (exactly equal in both lists) elements divided by the number of total unique elements.
    """
    sg = set(ground.split(delimeter))
    ss = set(sample.split(delimeter))
    common = sg & ss
    total = sg | ss
    return len(common) / len(total)

def ask_llm_to_compare(ground: str, sample: str, gen_handle: Callable[[str], str]) -> float:

    rating_scale = ["unrelated", "somewhat related", "similar", "very similar", "equivalent"]
    rs_with_formatting = [f"[[[{i}]]]" for i in rating_scale]
    flip = random() < 0.5 # prevent text order bias

    prompt = f"""
        You are a skilled text evaluator capable of comparing any two texts and rate their similarity.
        Compare these two texts along with whatever conclusions they come to. 

        <text1>
        {ground if flip else sample}
        </text1>
        <text2>
        {sample if flip else ground}
        </text2>

        Follow your explanations with your final verdict. Choose one option from the following: {rs_with_formatting}. Do not forget the three square brackets.
    """

    ans = gen_handle(prompt)
    verdict = parse_verdict(ans)
    if len(verdict) != 1 or verdict[0] not in rating_scale:
        print(f"WARNING: LLM Evaluator answer {ans} not part of the scale.")
        return 0.0
    
    return 0.25*rating_scale.index(verdict[0])

