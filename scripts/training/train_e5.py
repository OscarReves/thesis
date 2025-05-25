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

    # # Prepare training examples for being in memory 
    # train_examples = [
    #     InputExample(texts=[f"query: {q}", f"passage: {p}"])
    #     for q, p in tqdm(zip(dataset["query"], dataset["text"]), total=len(dataset), desc="Building training examples")
    # ]

    # # Save train examples
    # import pickle
    # with open("data/train_examples.pkl", "wb") as f:
    #     pickle.dump(train_examples, f)

    tokenized_path = 'data/training/tokenized_e5_inputs.pt'
    batch_size = 1024  # adjust based on your RAM

    if os.path.exists(tokenized_path):
        print(f"Loading pre-tokenized data from {tokenized_path}")
        tokenized = torch.load(tokenized_path)
    else:
        print(f"Tokenizing data in batches and saving to {tokenized_path}")
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large",use_fast=True)

        def tokenize_batch(batch):
            return tokenizer(
                [f"query: {q}" for q in batch["query"]],
                [f"passage: {p}" for p in batch["text"]],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1024,
            num_proc=16,  # adjust to core count
            remove_columns=dataset.column_names
        )
        # Concatenate into single tensors
        tokenized = {
            "input_ids": torch.tensor(tokenized_dataset["input_ids"]),
            "attention_mask": torch.tensor(tokenized_dataset["attention_mask"])
        }
        torch.save(tokenized, tokenized_path)


    device = torch.device("cuda")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(device)
    model.train()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.cuda.amp.GradScaler()  # for mixed precision

    # Load pre-tokenized data
    dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / count


    # Logger 
    writer = SummaryWriter(log_dir="./logs/e5_finetune")

    global_step = 0

    for epoch in range(1):  # add more epochs as needed
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for input_ids, attention_mask in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
                pooled = F.normalize(pooled, p=2, dim=1)

                # Similarity matrix (cosine sim)
                sims = pooled @ pooled.T  # [batch_size x batch_size]
                labels = torch.arange(len(sims), device=device)
                loss = F.cross_entropy(sims, labels)

            writer.add_scalar("train/loss", loss.item(), global_step)
            if global_step % 10 == 0:
                writer.flush()
            global_step += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())


    writer.close()

if __name__ == "__main__":
    main()
