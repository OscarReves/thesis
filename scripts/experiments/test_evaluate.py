import yaml
import argparse
from src.utils import load_documents
from src.evaluator import get_evaluator
from src.pipeline import evaluate_answers
from pathlib import Path

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # evaluates every open-domain answer in a directory
    directory = Path(config['directory'])
    answers_dir  = directory / 'open_domain'
    save_dir = directory / 'open_domain_evaluation'
    evaluator_name = 'gemma-9b-binary'
    batch_size = config['batch_size']
    evaluator = get_evaluator(evaluator_name)

    # iterate through directory 
    answers_directory = Path(answers_dir)

    for file in answers_directory.iterdir():
        answers_path = str(file)
        answers = load_documents(answers_path)

        save_path = save_dir / file.name

        evaluate_answers(
            answer_dataset = answers,
            evaluator = evaluator,
            save_path = save_path,
            batch_size=batch_size
            )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
