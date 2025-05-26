import yaml
from src.utils import load_retrieval_corpus
from src.retriever import get_retriever
from src.embedder import get_embedder
from src.generator import get_generator
from src.indexer import FaissIndexer
from src import pipeline as pipeline_module
from src.utils import save_to_json, load_documents, get_retrieval_accuracy, load_web_faq
import argparse 
from tqdm import tqdm
import os 
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
torch.backends.cudnn.benchmark = True


def main():

    # Build dataloader
    dataset_path = '/dtu/p1/oscrev/webfaq_danish'
    dataset = load_web_faq(dataset_path)

    tokenized_path = 'data/training/tokenized_e5_inputs.pt'
    batch_size = 1024  # adjust based on your RAM

    device = torch.device("cuda")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    model.train()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler(device='cuda')  # for mixed precision

    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / count


    # Logger 
    writer = SummaryWriter(log_dir="./logs/e5_finetune")

    global_step = 0

    # Tokenize queries with progress
    # add loading from disk in the future
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large",use_fast=True)
    queries = [f"query: {q}" for q in dataset["query"]]
    query_inputs = {"input_ids": [], "attention_mask": []}
    for q in tqdm(queries, desc="Tokenizing queries"):
        encoded = tokenizer(q, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        query_inputs["input_ids"].append(encoded["input_ids"])
        query_inputs["attention_mask"].append(encoded["attention_mask"])
    query_inputs["input_ids"] = torch.cat(query_inputs["input_ids"])
    query_inputs["attention_mask"] = torch.cat(query_inputs["attention_mask"])
    torch.save(query_inputs, "data/training/query_inputs.pt")

    # Tokenize passages with progress
    # add loading from disk in the future
    passages = [f"passage: {p}" for p in dataset["text"]]
    passage_inputs = {"input_ids": [], "attention_mask": []}
    for p in tqdm(passages, desc="Tokenizing passages"):
        encoded = tokenizer(p, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        passage_inputs["input_ids"].append(encoded["input_ids"])
        passage_inputs["attention_mask"].append(encoded["attention_mask"])
    passage_inputs["input_ids"] = torch.cat(passage_inputs["input_ids"])
    passage_inputs["attention_mask"] = torch.cat(passage_inputs["attention_mask"])
    torch.save(passage_inputs, "data/training/passage_inputs.pt")

    # Zip and load into DataLoader
    tensor_dataset = TensorDataset(
        query_inputs["input_ids"],
        query_inputs["attention_mask"],
        passage_inputs["input_ids"],
        passage_inputs["attention_mask"]
    )
    dataloader = DataLoader(tensor_dataset, batch_size=128, shuffle=True, num_workers=4)

    def mean_pooling(last_hidden, mask):
        mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def contrastive_loss(q, p, temperature=0.05):
        q = F.normalize(q, dim=1)
        p = F.normalize(p, dim=1)
        logits = torch.matmul(q, p.T) / temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    for epoch in range(8):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for q_ids, q_mask, p_ids, p_mask in pbar:
            q_ids, q_mask = q_ids.to(device), q_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)

            with torch.cuda.amp.autocast():
                q_out = model(input_ids=q_ids, attention_mask=q_mask)
                p_out = model(input_ids=p_ids, attention_mask=p_mask)

                q_emb = mean_pooling(q_out.last_hidden_state, q_mask)
                p_emb = mean_pooling(p_out.last_hidden_state, p_mask)

                loss = contrastive_loss(q_emb, p_emb)

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

        save_path = f"models/e5_finetuned_epoch{epoch}.pt"
        torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
