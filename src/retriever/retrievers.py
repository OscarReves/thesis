from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from datasets import load_from_disk
import torch.nn.functional as F

class GPT2Retriever:
    def __init__(self, index_path, documents, device=None, text_field='text'):
        model_name = 'gpt2'
        self.device = torch.device(device)

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load dataset and extract text field
        self.dataset = documents  # or load_dataset(...)
        self.contexts = self.dataset[text_field]      # list of texts
        self.titles = self.dataset['id']

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
            return F.normalize(pooled, p=2, dim=1).cpu().numpy()
    
    def retrieve(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        return [
            # consider returning a list instead and joining somewhere else
            # likewise, consider mapping the index to documents with an ID
            "\n\n".join([self.contexts[idx] for idx in indices])
            for indices in I
        ]
    
    def retrieve_titles(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        return [
            # consider returning a list instead and joining somewhere else
            # likewise, consider mapping the index to documents with an ID
            [self.titles[idx] for idx in indices]
            for indices in I
        ]        

class E5Retriever:
    def __init__(self, index_path, documents, device=None, text_field='text'):
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.device = torch.device(device)

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load dataset and extract text field
        self.dataset = documents  # or load_dataset(...)
        self.contexts = self.dataset[text_field]      # list of texts
        self.titles = self.dataset['id']

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
            return F.normalize(pooled, p=2, dim=1).cpu().numpy()
    
    def retrieve(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        return [
            # consider returning a list instead and joining somewhere else
            # likewise, consider mapping the index to documents with an ID
            "\n\n".join([self.contexts[idx] for idx in indices])
            for indices in I
        ]
    
    def retrieve_titles(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        return [
            # consider returning a list instead and joining somewhere else
            # likewise, consider mapping the index to documents with an ID
            [self.titles[idx] for idx in indices]
            for indices in I
        ]        
