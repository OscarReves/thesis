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
#torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def main():

    # 1) load the dataset
    dataset_path = '/dtu/p1/oscrev/webfaq_danish'
    ds = load_web_faq(dataset_path)

    # 2) split into train/validation
    splits     = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = splits["train"]
    val_ds   = splits["test"]

    # 3) Build (query, passage) tuples with tqdm
    webfaq_train_pairs = list(
        tqdm(zip(train_ds["query"], train_ds["text"]),
            total=len(train_ds),
            desc="Building train_pairs")
    )
    webfaq_val_pairs = list(
        tqdm(zip(val_ds["query"], val_ds["text"]),
            total=len(val_ds),
            desc="Building val_pairs")
    )

    # 4) Load your pretrained E5 model
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # 5) Prepare your Danish WebFAQ data as InputExample pairs
    #    Each example is a (query, positive_passage) pair
    train_examples = [
        InputExample(texts=[f"query: {q}", f"passage: {p}"])
        for q, p in tqdm(
            webfaq_train_pairs,
            total=len(webfaq_train_pairs),
            desc="Building train_examples"
        )
    ]

    # 6) Wrap in a DataLoader
    train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)

    # 7) Choose a contrastive loss (MultipleNegativesRankingLoss ≈ InfoNCE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 8) Prepare val InputExamples and IR‐evaluator
    #    We need query‐to‐passage mapping for the evaluator, so we split
    val_queries = [q for q, p in webfaq_val_pairs]
    val_corpus  = list({p for _, p in webfaq_val_pairs})
    val_qrels   = { i: [val_corpus.index(p)] 
                    for i, (_, p) in enumerate(webfaq_val_pairs) }

    val_evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_qrels,          # exactly this dict: {q_idx: [p_idx], …}
        batch_size=64,
        name="webfaq-val",
        show_progress_bar=True
    )
    # 10) Fine-tune with evaluation every 500 steps and save best model by val MRR
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        evaluation_steps=500,
        epochs=3,
        warmup_steps=200,
        output_path="./e5-danish-finetuned",
        save_best_model=True
    )

if __name__ == "__main__":
    main()
