import yaml
import argparse
from src.utils import load_documents
from src.evaluator import get_evaluator
from src.pipeline import evaluate_answers

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    answers_path = config['answers_path']
    evaluator_name = config['evaluator_name']
    save_path = config['evaluation_path']
    batch_size = config['batch_size']

    answers = load_documents(answers_path)
    evaluator = get_evaluator(evaluator_name)

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
