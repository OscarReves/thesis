from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np

class GPT2Embedder:
    def __init__(self, device, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device)

    def encode(self, documents, batch_size=32):
        device = self.device
        self.model.eval()
        self.model.to(device)

        texts = [f"passage: {t}" for t in documents['body']]
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                output = self.model(**tokens).last_hidden_state
                attention_mask = tokens["attention_mask"].unsqueeze(-1).expand(output.size())
                summed = torch.sum(output * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = summed / counts
                all_embeddings.extend(embeddings.cpu().numpy())

        return np.array(all_embeddings, dtype=np.float32)

class E5Embedder:
    def __init__(self, device, model_name="intfloat/multilingual-e5-large-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device)

    def encode(self, documents, batch_size=32):
        device = self.device
        self.model.eval()
        self.model.to(device)

        texts = [f"passage: {t}" for t in documents['body']]
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                output = self.model(**tokens).last_hidden_state
                attention_mask = tokens["attention_mask"].unsqueeze(-1).expand(output.size())
                summed = torch.sum(output * attention_mask, dim=1) 
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = summed / counts # mean pooling
                all_embeddings.extend(embeddings.cpu().numpy())

        return np.array(all_embeddings, dtype=np.float32)


