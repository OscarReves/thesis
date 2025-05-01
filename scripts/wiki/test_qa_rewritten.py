import yaml
from src.utils import load_documents_from_directory, load_squad, load_squad_rewritten
from src.retriever import get_retriever
from src.generator import get_generator
from src.pipeline import test_qa_with_retrieval_wiki
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
    save_path = config['qa_save_path']
    generator_name = config['generator_name']
    max_samples = config['n_questions']
    batch_size = config['eval_batch_size']
    questions_with_title = config['squad_questions_with_title']

    print("Loading documents...")
    documents = load_documents_from_directory(documents_path)
    
    print("Loading questions...")
    question_dataset = load_squad_rewritten(questions_path, prepend_with_title=questions_with_title)
    
    print("Loading retriever...")
    retriever = get_retriever(
        retriever_name,
        documents = documents,
        index_path = index_path,
        device = device
        )
    
    print("Loading generator...")
    generator = get_generator(generator_name)

    print("Testing qa with retrieval...")
    test_qa_with_retrieval_wiki(
        question_dataset = question_dataset, 
        retriever = retriever, 
        generator = generator,
        save_path = save_path,
        max_samples=max_samples,
        batch_size=batch_size,
        silent=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
