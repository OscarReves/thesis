import yaml
from src.utils import load_documents_from_directory, load_questions
from src.retriever import get_retriever
from src.pipeline import test_retrieval_with_uid
import argparse 
import faiss
import numpy as np


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    questions_path = config['questions_path']
    documents_path = config['chunked_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_path = config['retrieval_save_path']
    
    documents = load_documents_from_directory(documents_path)
    question_dataset = load_questions(questions_path)

    index = faiss.read_index(index_path)
    print(index.ntotal)
    print(type(index))  # Should be IndexIDMap or similar if using add_with_ids

    retriever = get_retriever(
         retriever_name,
         documents = documents,
         index_path = index_path,
         device = device
         )

    test_retrieval_with_uid(
        question_dataset = question_dataset,
        retriever = retriever,
        save_path = save_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)


