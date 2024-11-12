from typing import Callable
from utils import parse_verdict, parse_answer
from random import random
import re
import io
import multiprocessing
from contextlib import redirect_stdout
def ask_llm_to_compare(ground: str, sample: str, gen_handle: Callable[[str], str]) -> float:
    """
    Ask llm to rate the similarity of the two solutions based on a 5-point scale.
    """

    rating_scale = ["unrelated", "somewhat related", "similar", "very similar", "equivalent"]
    rs_with_formatting = [f"[[[{i}]]]" for i in rating_scale]
    flip = random() < 0.5 # prevent text order bias

    prompt = f"""
        You are a skilled text evaluator capable of comparing any two texts and rate their semantic similarity.
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

def binary_match(ground: str, sample: str) -> float:
    results = parse_answer(sample)
    for r in results:
        if r.strip() == ground.strip():
            return 1.0
    return 0.0


def my_exec(input_data, code, queue):
    f = io.StringIO()
    inp = iter(input_data)
    local_vars = {"input": lambda: next(inp)}
    with redirect_stdout(f):
        exec(code, {}, local_vars)
    # Send the captured output back to the main process
    queue.put(f.getvalue().strip())

def run_code(test_cases: dict[str, list[str]], tested_code: str) -> float:
    test_input = test_cases['test_inputs']
    test_output = test_cases['test_outputs']
    # if LLM used markdown despite being asked not to
    parts = tested_code.split('```')
    if len(parts) == 3:
        tested_code = '\n'.join(parts[1].split('\n')[1:])
    
    res = []
    for i in range(len(test_input)):
        score = 0
        queue = multiprocessing.Queue()  # Queue to capture output
        try:
            # Set up a separate process for code execution
            p = multiprocessing.Process(target=my_exec, args=(test_input[i], tested_code, queue))
            p.start()
            p.join(3)  # Wait for 3 seconds

            if p.is_alive():
                # If process is still running, terminate it
                p.terminate()
                print("Process exceeded time limit.")
                score += 0.05
                captured_output = "" 
            else:
                score += 0.1 
                captured_output = queue.get() if not queue.empty() else ""
            
        except Exception as e:
            print(f"Unexpected exception occurred: {e}")
            captured_output = ""

        if captured_output == test_output[i]:
            score += 0.9
        res.append(score)
    
    # Full score only if all tests are passed
    if all([x == 1.0 for x in res]):
        return 1.0
    return 0.75 * sum(res) / len(res)