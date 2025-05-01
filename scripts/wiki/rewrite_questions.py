import yaml
from src.utils import load_documents_from_directory, load_squad
from src.retriever import get_retriever
from src.generator import get_generator
from src.pipeline import rewrite_questions
import argparse 

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # in the future clean this up with the double asteriks ** method
    questions_path = config['questions_path']
    device = config['device']
    generator_name = config['generator_name']
    max_samples = config['n_questions']
    batch_size = config['batch_size']
    questions_with_title = config['squad_questions_with_title']
    save_path = config['save_path']
    
    print("Loading questions...")
    question_dataset = load_squad(questions_path, prepend_with_title=questions_with_title)
    
    print("Loading generator...")
    generator = get_generator(generator_name)

    print("Rewriting questions")
    rewrite_questions(
        question_dataset=question_dataset,
        generator=generator,
        save_path=save_path,
        batch_size=batch_size,
        max_samples=max_samples
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
