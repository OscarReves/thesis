from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from huggingface_hub import login, whoami
from dotenv import load_dotenv
import os

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

        texts = [f"passage: {t}" for t in documents['text']]
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
    def __init__(self, device='cuda', model_name='intfloat/multilingual-e5-large-instruct'):
        if 'cuda' in device and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available on this system (device={device})")
        
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print(f"Loaded model {model_name} as embedder")
        self.model.eval()
        torch.set_float32_matmul_precision('high') # is this necessary?

    def encode(self, documents, batch_size=64):  # bump batch size if your GPU allows
        texts = [f"passage: {t}" for t in documents['text']] # for very large batch sizes this is likely ineffecient. Consider .map 
        all_embeddings = []

        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc=
                          f"Encoding chunks in batches of {batch_size}"):
                batch = texts[i:i+batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")#.to(self.device)
                tokens = {k: v.to(self.device, non_blocking=True) for k, v in tokens.items()}
                output = self.model(**tokens).last_hidden_state

                mask = tokens['attention_mask'].unsqueeze(-1)
                summed = torch.sum(output * mask, dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts

                all_embeddings.append(embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()

        return embeddings

    def encode_pretokenized(self, path, batch_size=64):
        tokenized = torch.load(path)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)  # or just attention_mask.sum(1)
        max_len = lengths.max().item()
        print(f"Max token length: {max_len}")
        
        all_embeddings = []
        with torch.inference_mode():
            for i in range(0, len(input_ids), batch_size):
                batch_ids = input_ids[i:i+batch_size].to(self.device)
                batch_mask = attention_mask[i:i+batch_size].to(self.device)

                output = self.model(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state

                # Mean pooling
                mask = batch_mask.unsqueeze(-1)
                summed = torch.sum(output * mask, dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts

                all_embeddings.append(embeddings)

        embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()



    def encode_query(self, queries, batch_size=64):  # bump batch size if your GPU allows
        #texts = [f"query: {t}" for t in documents['query']] # for very large batch sizes this is likely ineffecient. Consider .map 
        queries = [f"query: {q}" for q in queries] # assume iterable 
        # all_embeddings = []

        # no reason to batch this as queries are usually passed in batches already
        with torch.inference_mode():
            # for i in tqdm(range(0, len(queries), batch_size), desc=
            #               f"Encoding chunks in batches of {batch_size}"):
            #     batch = queries[i:i+batch_size]
            tokens = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(self.device)
            output = self.model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            summed = torch.sum(output * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts

            # all_embeddings.append(embeddings)

        # embeddings = torch.cat(all_embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()

        return embeddings

class E5EmbedderLocal(E5Embedder):
    def __init__(self, 
                    device='cuda', 
                    model_name="intfloat/multilingual-e5-large",
                    save_path='models/e5_finetuned_epoch7.pt'
            ):
        #super().__init__(device, model_name)
        if 'cuda' in device and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available on this system (device={device})")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-large",torch_dtype=torch.float16).to(device)
        # load weights of saved model
        print(f"Loading state dict from {save_path}")
        state_dict = torch.load(save_path)
        # because of dual-gpu training, state_dict needs to be refactored 
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} 
        self.model.load_state_dict(new_state_dict)
        #model.to("cuda")
        self.model.eval()
        #torch.set_float32_matmul_precision('high') # is this necessary?


class E5Finetuned(E5Embedder):
    def __init__(self, device='cuda', model_name='coffeecat69/e5_finetuned_epoch2'):
        super().__init__(device, model_name='coffeecat69/e5_finetuned_epoch2')

class E5Large(E5Embedder):
    def __init__(self, device='cuda', model_name='intfloat/multilingual-e5-large'):
        super().__init__(device, model_name)

class BertTinyEmbedder:
    def __init__(self, device, model_name='prajjwal1/bert-tiny'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(device)

    def encode(self, documents, batch_size=32):
        device = self.device
        self.model.eval()

        texts = [f"passage: {t}" for t in documents['text']]
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
                ).to(device)

            with torch.no_grad():
                output = self.model(**tokens).last_hidden_state
                cls_embeddings = output[:, 0, :]
                all_embeddings.extend(cls_embeddings.cpu().numpy())

        return np.array(all_embeddings, dtype=np.float32)

