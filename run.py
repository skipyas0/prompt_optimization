from typing import Callable, Generic
from evolutionary import EvolutionaryAlgorithm, EvoParams
from evaluators import simple_list_intersection
from evoprompt import Prompt
from vllm_api import OpenAIPredictor
from datetime import datetime
import random 

if __name__ == "__main__":
    rel_path = "tasks/entity_extraction/"
    task_name = "german-intelligence-says-russian-gru-group-behind-nato-eu-cyberattacks-2024-09-09"

    with open(f"{rel_path}/generation.txt") as f:
        gen_instructions = "Hello world! This is a testy test."#f.read()
    with open(f"{rel_path}/samples/x/{task_name}.txt") as f:
        task = "Generic task :O. What could it be?"#f.read()
    with open(f"{rel_path}/samples/y/{task_name}.txt") as f:
        ground = f.read()

    model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    # "mistralai/Mistral-7B-v0.3"
    #api = OpenAIPredictor(model)
    #prelude = [{"role": "system", "message": "You are a helpful assistant. You follow instructions and answer concisely."}]
    #gen_handle = lambda x: api.predict(prelude, x)

    #score = lambda x: simple_list_intersection(ground.split(','), x)

    def scramble(input: str) -> str:
        char_list = list(input)
        random.shuffle(char_list)
        return ''.join(char_list)
    gen_handle = scramble

    log_file = f"logs/{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}.txt"

    class MyPrompt(Prompt):
        def __init__(self, prompt: str) -> None:
            super().__init__(gen_handle, prompt)
            self.result = ""

        def log(self) -> None:
            with open(log_file, 'a') as f:
                f.write(f"Prompt gen: {self.generation}\n\n***\n{self.traits[0]}\n***\nResult:\n{self.result}\n\nScore: {self.fitness}\n\n######")

        def _fitness(self) -> float:
            self.result = self.generate(self.traits[0] + task)
            res = random.random()#score(self.result.split(','))
            self.fitness = res
            self.log()
            return res
    
    class MyEA(EvolutionaryAlgorithm):
        def __init__(self, params: EvoParams) -> None:
            super().__init__(params)

        def populate(self) -> None:
            for _ in range(self.params.initial_population_size):
                prompt = gen_handle(gen_instructions)
                self.population.append(MyPrompt(prompt))

    params = EvoParams()
    EA = MyEA(params)
    EA.populate()
    EA.run()
