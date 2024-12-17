import utils
import multiprocessing
from rouge_score import rouge_scorer
import os

def logging_wrapper(test_sample, answer, func = None) -> float:
    out = func(test_sample, answer)
    ground = test_sample["answer"] if "answer" in test_sample.keys() else ""
    utils.log_usage(os.getenv("CALL_LOG_FILE"), (ground, answer), out)
    return out


def rouge_simple(test_sample, answer) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(test_sample['answer'], answer)["rougeL"].fmeasure

def rouge_diff(test_sample, answer) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    baseline = scorer.score(test_sample['question'], test_sample['answer'])["rougeL"].fmeasure
    ans_score = scorer.score(answer, test_sample['answer'])["rougeL"].fmeasure
    return 3-2*(baseline/ans_score)**3  


def binary_match(test_sample, answer) -> float:
    ground = test_sample['answer']
    results = utils.parse_answer(answer)
    for r in results:
        if r.strip() == ground.strip():
            return 1.0
    return 0.0

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
            p = multiprocessing.Process(target=utils.my_exec, args=(test_input[i], tested_code, queue))
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

ff_dict_no_log = {
    "rouge_simple": rouge_simple,
    "rouge_diff": rouge_diff,
    "binary_match": binary_match,
    "run_code": run_code
}
ff_dict = {k: (lambda a, b, func = v: logging_wrapper(a, b, func = func)) for k, v in ff_dict_no_log.items()}