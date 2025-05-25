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

class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class PreTokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return InputFeatures(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx]
        )

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

    #train_dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    train_dataset = PreTokenizedDataset(tokenized["input_ids"], tokenized["attention_mask"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,            # try 128–256 on H100 with use_amp=True
        shuffle=True,
        num_workers=32,            # plenty of cores available
git         pin_memory=True,
        prefetch_factor=4
    )
    
    # load model
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Loss: in-batch negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Logger 
    writer = SummaryWriter(log_dir="./logs/e5_finetune")
    def log_callback(score, epoch, step):
        writer.add_scalar("training/loss", score, step)
        writer.flush() 

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=500,
        use_amp=True,
        output_path="models/e5-finetuned",
        callback=log_callback
    )
    writer.close()

if __name__ == "__main__":
    main()
