import yaml
from src.utils import load_squad, load_documents, save_squad_contexts
from src.retriever import get_retriever
from src.embedder import get_embedder
from src.pipeline import test_batched_retrieval_with_uid, test_retrieval, test_retrieval_with_uid
from src.indexer import FaissIndexer
import argparse 
import faiss
import numpy as np


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    squad_path = config['squad_path']
    squad_context_path = config['squad_context_path']
    questions_path = config['questions_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_path = config['retrieval_save_path']
    batch_size = config['batch_size']
    max_samples = config['n_questions']
    questions_with_title = config['squad_questions_with_title']
    embedder_name = config['embedder_name']
    
    print(f"Preprocessing SQuAD contexts...")
    save_squad_contexts(load_path=squad_path, save_path=squad_context_path)

    print(f"Initializing embedder ({embedder_name})...", flush=True)
    embedder = get_embedder(name = embedder_name, device = device)
    
    print("Loading indexer...")
    indexer = FaissIndexer(embedder=embedder, index_path=index_path)
    
    documents = load_documents(squad_context_path)

    print("Indexing documents...")
    indexer.index_documents_with_uid(documents, batch_size=batch_size)

    print(f"Index built and saved to {index_path}")

    question_dataset = load_squad(questions_path, prepend_with_title=questions_with_title, with_context=True)

    retriever = get_retriever(
         retriever_name,
         documents = documents,
         index_path = index_path,
         device = device
         )

    test_batched_retrieval_with_uid(
        question_dataset = question_dataset,
        retriever = retriever,
        save_path = save_path,
        batch_size=batch_size
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)


