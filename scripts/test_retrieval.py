import yaml
from src.utils import load_documents, load_questions
from src.retriever import get_retriever
from src.pipeline import test_retrieval
import argparse 

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    questions_path = config['questions_path']
    documents_path = config['chunked_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_path = config['save_path']
    
    documents = load_documents(documents_path)
    question_dataset = load_questions(questions_path)
    retriever = get_retriever(
        retriever_name,
        documents = documents,
        index_path = index_path,
        device = device
        )

    test_retrieval(question_dataset, retriever, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)


