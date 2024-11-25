import openai
import os
from dotenv import load_dotenv
from stats import stats
import utils

class ModelAPI(utils.FromJSON):
    default_path = "conf/_model/{}.json"
    def __init__(self, model: str, sol_temp: float, meta_temp: float, name: str) -> None:
        self.model = model
        self.sol_temp = sol_temp
        self.meta_temp = meta_temp
        self.name = name
        if self.model == "gpt-4o-mini":
            load_dotenv()
            self.client = openai.OpenAI(
                api_key=os.getenv("API_KEY"),
            )
        elif self.model == "debug":
                self.predict = utils.scramble
        else:
            port = os.getenv("VLLM_MY_PORT")
            self.client = openai.OpenAI(
                base_url=f"http://localhost:{port}/v1",
                api_key="EMPTY",
            )

        self.solve = lambda q: self.predict(q, "solve")
        self.generate = lambda q: self.predict(q, "generate")
            
    def forward(self, messages, temp):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
        )
        return completion
    
    def predict(self, question, context, messages=None) -> str:
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant. You follow instructions and answer concisely."}]
        msgs = messages + [{"role": "user", "content": question}]


        try:
            completion = self.forward(
                messages=msgs,
                temp=self.sol_temp if context == "solve" else self.meta_temp
            )
            #print(completion)
        except openai.BadRequestError as e:
            print(f"WARNING: openai.BadRequestError for: {msgs}")
            completion = {"error": "openai.BadRequestError", "choices": [{"message": {"content": str(e)}}]}
        
        stats.add_to_current_step({"Total LLM calls": 1})
        stats.append_to_current_step({
            f"{context} tokens in": completion.usage.prompt_tokens,
            f"{context} tokens out": completion.usage.completion_tokens
        })
        return completion.choices[0].message.content
    