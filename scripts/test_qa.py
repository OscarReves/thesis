import yaml
from src.utils import load_documents, load_questions
from src.retriever import get_retriever
from src.generator import get_generator
from src.pipeline import test_qa_with_retrieval
import argparse 

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # in the future clean this up with the double asteriks ** method
    questions_path = config['questions_path']
    documents_path = config['chunked_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_path = config['save_path']
    generator_name = config['generator_name']

    documents = load_documents(documents_path)
    question_dataset = load_questions(questions_path)
    retriever = get_retriever(
        retriever_name,
        documents = documents,
        index_path = index_path,
        device = device
        )
    generator = get_generator(generator_name)

    test_qa_with_retrieval(
        question_dataset = question_dataset, 
        retriever = retriever, 
        generator = generator,
        save_path = save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
