from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

class Bert():
    def __init__(self, model_name: str = 'bert-base-uncased') -> None:
        # Load pre-trained BERT model and tokenizer
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_bert_embedding(self, text):
        # Tokenize and encode the text
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Get the hidden states (embeddings) from BERT
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the last hidden state of the [CLS] token (which represents the sentence embedding)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return cls_embedding

    def bert_cosine_similarity(self, text1, text2):
        # Get the embeddings for both texts
        embedding1 = self.get_bert_embedding(text1)
        embedding2 = self.get_bert_embedding(text2)
        
        # Compute cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        
        return similarity
    
    def cos_sim_precalc(self, embed1, embed2):
        return cosine_similarity([embed1], [embed2])[0][0]