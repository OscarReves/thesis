import yaml
import argparse
from src.utils import load_documents, get_accuracy

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    answers_path = config['evaluation_path']
    answers = load_documents(answers_path)
    accuracy = get_accuracy(answers, type='multiple_choice')
    print(f" {answers_path} Accuracy: {accuracy:.2f}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
