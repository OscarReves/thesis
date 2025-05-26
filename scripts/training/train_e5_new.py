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

    # 1) load the Danish config
    dataset_path = '/dtu/p1/oscrev/webfaq_danish'
    ds = load_web_faq(dataset_path)
    qrels   = ds["qrels"]    # has "query-id" & "corpus-id"
    queries = ds["queries"]  # has "_id" & "text"
    corpus  = ds["corpus"]   # has "_id" & "text"

    # 2) rename to align keys
    queries = queries.rename_column("_id",    "query_id")  \
                    .rename_column("text",   "query")
    corpus  = corpus.rename_column("_id",     "corpus_id") \
                    .rename_column("text",    "passage")
    qrels   = qrels.rename_column("query-id",  "query_id")   \
                .rename_column("corpus-id", "corpus_id")

    # 3) split qrels into train/validation
    splits     = qrels.train_test_split(test_size=0.1, seed=42)
    train_qrels = splits["train"]
    val_qrels   = splits["test"]

    # 4) join onto queries & corpus for each split
    def make_pairs(qr):
        joined = qr.join(queries, on="query_id") \
                .join(corpus,  on="corpus_id")
        return list(zip(joined["query"], joined["passage"]))

    webfaq_train_pairs = make_pairs(train_qrels)
    webfaq_val_pairs   = make_pairs(val_qrels)


    # 1. Load your pretrained E5 model
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # 2. Prepare your Danish WebFAQ data as InputExample pairs
    #    Each example is a (query, positive_passage) pair
    
    train_examples = [
        InputExample(texts=[f"query: {q}", f"passage: {p}"])
        for q, p in tqdm(
            zip(webfaq_train_pairs["query"], webfaq_train_pairs["passage"]),
            total=len(webfaq_train_pairs),
            desc="Building train_examples"
        )
    ]

    # 3. Wrap in a DataLoader
    train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)

    # 4. Choose a contrastive loss (MultipleNegativesRankingLoss ≈ InfoNCE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 2) Prepare val InputExamples and IR‐evaluator
    #    We need query‐to‐passage mapping for the evaluator, so we split
    val_queries, val_pos = zip(*webfaq_val_pairs)
    val_corpus = list(set(val_pos))  # all unique passages in your val split
    val_qrels = { i: [val_corpus.index(p)] for i,p in enumerate(val_pos) }

    # DataLoader just for batching the queries & corpus
    val_evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        query_ids=list(range(len(val_queries))),
        relevant_docs=val_qrels,
        batch_size=64,
        name="webfaq-val"
    )

    # 3) Fine-tune with evaluation every 500 steps and save best model by val MRR
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=3,
        evaluation_steps=500,     # run eval every 500 training steps
        warmup_steps=200,
        output_path="models/e5_new",
        save_best_model=True
    )

if __name__ == "__main__":
    main()
