import yaml
from src.utils import load_knowledge_base, load_questions_by_type
from src.retriever import get_retriever
from src.generator import get_generator
from src import pipeline as pipeline_module
import argparse 

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # in the future clean this up with the double asteriks ** method
    questions_path = config['questions_path']
    kb_path = config['kb_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_path = config['save_path']
    max_samples = config['n_questions']
    batch_size = config['batch_size']
    kb_type = config['kb_type']
    question_type = config['question_type']
    pipeline_name = 'test_retrieval'


    if kb_path:
        print("Loading knowledge base...")
        documents = load_knowledge_base(kb_path, kb_type) # is now abstracted 
    else:
        print("No knowledge base specified...")
        documents = None

    print("Loading questions...")
    question_dataset = load_questions_by_type(questions_path, type = question_type)

    if documents:
        print("Loading retriever...")
        retriever = get_retriever(
            retriever_name,
            documents = documents,
            index_path = index_path,
            device = device
            )
    else:
        retriever = None

    #print("Available functions in pipeline_module:", dir(pipeline_module))
    #print("Trying to access:", pipeline_name)
    pipeline_func = getattr(pipeline_module, pipeline_name)
    pipeline_func(
        question_dataset = question_dataset, 
        retriever = retriever, 
        save_path = save_path,
        max_samples=max_samples,
        batch_size=batch_size,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
