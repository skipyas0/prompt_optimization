from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2Tokenizer

class LocalPredictor:
    def __init__(self, model, temp=1.0):
        self.model = model
        self.temp = temp
        """ model = AutoModelForCausalLM.from_pretrained(
            model, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        """
        self.args = {
            "max_new_tokens": 50,
            "temperature": temp,
        } 

        
        self.pipeline= pipeline('text-generation', model="openai-community/gpt2", device='cuda')
   
    def predict(self, messages=None, question=None) -> str:
        if messages is None:
            msg = f'system: You are a helpful assistant, you answer consisely.\nuser: {question}\nassistant:'
        else:
            msg = f'{messages[0]["role"]}: {messages[0]["content"]}\nuser: {question}\nresponse:'
        return self.pipeline(msg, **self.args)[0]['generated_text']
    
if __name__ == "__main__":
    model = LocalPredictor("openai-community/gpt2")

    prelude = {"role": "system", "content": "You are a wandering wizard who only responds with archaic riddles."}
    question = "SolidGoldMagikarp"

    c = model.predict(None, question)

    print(c)