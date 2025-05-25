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


def main():
    
    # Build dataloader
    documents_path = '/dtu/p1/oscrev/webfaq_danish'
    documents = load_web_faq(documents_path)
    
    def to_input_example(example):
        return InputExample(texts=[f"query: {example['query']}", f"passage: {example['text']}"])

    # Assuming `documents` is a HuggingFace Dataset
    train_dataset = documents.map(lambda x: {"input_example": to_input_example(x)})

    # Extract list of InputExamples
    train_examples = train_dataset["input_example"]
    train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Loss: in-batch negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path="models/e5-finetuned"
    )

if __name__ == "__main__":
    main()
