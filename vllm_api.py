import openai
import os

class OpenAIPredictor:
    def __init__(self, model):
        self.model = model

        port = os.environ["VLLM_MY_PORT"]

        self.client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )

    def forward(self, messages, temperature, n):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            frequency_penalty=0.1
        )
        return completion
    
    def predict(self, messages, question=None, n=1, temperature=0.0):
        if question:
            msgs = messages + [{"role": "user", "content": question}]
        else:
            msgs = messages

        try:
            completion = self.forward(
                messages=msgs,
                temperature=temperature,
                n=n,
            )

        except openai.BadRequestError as e:
            print(f"WARNING: openai.BadRequestError for: {msgs}")
            completion = {"error": "openai.BadRequestError", "choices": [{"message": {"content": str(e)}}] * n}
        
        print(completion)
        return completion.choices[0].message.content
    
if __name__ == "__main__":
    model = OpenAIPredictor("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")

    prelude = {"role": "system", "content": "You are a wandering wizard who only responds with archaic riddles."}
    question = "Wow, you are handsome! What's your name?"

    c = model.predict([prelude], question)

    print(c)