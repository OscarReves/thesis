import yaml
from src.utils import load_knowledge_base, load_questions_by_type
from src.retriever import get_retriever
from src.generator import get_generator
from src import pipeline as pipeline_module
import argparse 
from pathlib import Path


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # SHOULD TEST QA FOR A SINGLE MODEL WITH/WIHTOUT CONTEXT 
    # AND BOTH OPEN-DOMAIN AND MC

    # in the future clean this up with the double asteriks ** method
    questions_path = config['questions_path']
    kb_path = config['kb_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    device = config['device']
    save_dir = config['save_path']
    generator_name = config['generator_name']
    max_samples = config['n_questions']
    batch_size = config['batch_size']
    kb_type = config['kb_type']
    question_type = config['question_type']
    pipelines = config.get('pipelines')
    silent = config.get('silent') # defaults to none 
    top_k = config.get('top_k', 5)

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
            device = device,
            top_k = top_k,
            )
    else:
        retriever = None

    print("Loading generator...")
    generator = get_generator(generator_name)


    # Loop through each pipeline function, dynamically building the save-paths 
    # retriever needs to have top_k dynamically adjusted 

    for suffix, pipeline in pipelines.items():
        # dynamically create the save path eg gemma_no_context
        file_name = generator_name + suffix
        folder = Path(save_dir)
        save_path = folder / file_name

        pipeline_name = pipeline[suffix]

        pipeline_func = getattr(pipeline_module, pipeline_name)
        pipeline_func(
            question_dataset = question_dataset, 
            retriever = retriever, 
            generator = generator,
            save_path = save_path,
            max_samples=max_samples,
            batch_size=batch_size,
            silent=silent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
