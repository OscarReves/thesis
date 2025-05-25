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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

def main():
    
    # Build dataloader
    dataset_path = '/dtu/p1/oscrev/webfaq_danish'
    dataset = load_web_faq(dataset_path)

    # Prepare training examples for being in memory 
    train_examples = [
        InputExample(texts=[f"query: {q}", f"passage: {p}"])
        for q, p in tqdm(zip(dataset["query"], dataset["text"]), total=len(dataset), desc="Building training examples")
    ]

    # Save train examples
    import pickle
    with open("data/train_examples.pkl", "wb") as f:
        pickle.dump(train_examples, f)

    train_dataloader = DataLoader(
        train_examples,
        batch_size=64,
        shuffle=True,
        num_workers=32,  # tune depending on CPU
        pin_memory=True
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

if __name__ == "__main__":
    main()
