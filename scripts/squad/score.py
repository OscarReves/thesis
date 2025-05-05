import yaml
import argparse
from src.utils import load_documents, get_retrieval_accuracy

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    answers_path = config['retrieval_save_path']
    answers = load_documents(answers_path)
    accuracy = get_retrieval_accuracy(answers)
    print(f"Accuracy: {accuracy:.2f}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
