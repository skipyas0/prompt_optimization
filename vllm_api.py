import openai
import os
from stats import stats
class OpenAIPredictor:
    def __init__(self, model, temp=0.75):
        self.model = model
        self.temp = temp
        port = os.environ["VLLM_MY_PORT"]

        self.client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )

    def forward(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temp,
            n=1,
            frequency_penalty=0.1
        )
        return completion
    
    def predict(self, messages=None, question=None) -> str:
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant. You follow instructions and answer concisely."}]
            
        if question:
            msgs = messages + [{"role": "user", "content": question}]
        else:
            msgs = messages

        try:
            completion = self.forward(
                messages=msgs,
            )

        except openai.BadRequestError as e:
            print(f"WARNING: openai.BadRequestError for: {msgs}")
            completion = {"error": "openai.BadRequestError", "choices": [{"message": {"content": str(e)}}]}
        
        stats.add_to_current_step({"Total LLM calls": 1})
        stats.append_to_current_step({
            "Tokens in": completion.usage.prompt_tokens,
            "Tokens out": completion.usage.completion_tokens
        })
        return completion.choices[0].message.content
    
if __name__ == "__main__":
    model = OpenAIPredictor("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")

    prelude = {"role": "system", "content": "You are a wandering wizard who only responds with archaic riddles."}
    question = "Wow, you are handsome! What's your name?"

    for _ in range(10):
        c = model.predict([prelude], question)

        print(c)