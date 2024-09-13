from typing import Callable
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
    prompt = f"""
        You are a skilled text evaluator capable of comparing any two texts and rate their alignment and similarity.
        Evaluate these two texts. Answer with one of the following: {rating_scale}.

        <text1>
        {ground}
        </text1>
        <text2>
        {sample}
        </text2>

        These two texts are:
    """

    ans = gen_handle(prompt).strip()
    if ans not in rating_scale:
        print(f"WARNING: LLM Evaluator answer {ans} not part of the scale.")
        return 0.0
    
    return 0.25*rating_scale.index(ans)

