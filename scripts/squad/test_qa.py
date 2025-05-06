import yaml
from src.utils import load_documents, load_squad
from src.retriever import get_retriever
from src.generator import get_generator
from src import pipeline as pipeline_module
import argparse 

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # in the future clean this up with the double asteriks ** method
    questions_path = config['questions_path']
    documents_path = config['contexts_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_path = config['qa_save_path']
    generator_name = config['generator_name']
    max_samples = config['n_questions']
    batch_size = config['batch_size']
    questions_with_title = config['squad_questions_with_title']
    pipeline_name = config.get('pipeline', 'test_qa_with_retrieval_wiki')


    print("Loading documents...")
    documents = load_documents(documents_path)
    
    print("Loading questions...")
    question_dataset = load_squad(questions_path, prepend_with_title=questions_with_title)
    
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
    pipeline_func = getattr(pipeline_module, pipeline_name)
    pipeline_func(
        question_dataset = question_dataset, 
        retriever = retriever, 
        generator = generator,
        save_path = save_path,
        max_samples=max_samples,
        batch_size=batch_size,
        silent=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
