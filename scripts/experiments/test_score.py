import yaml
import argparse
from src.utils import load_documents, get_accuracy
from pathlib import Path

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    directory = Path(config['directory'])
    mc_evaluation_path = config['save_dir']

    multiple_choice_dir = directory / mc_evaluation_path
    open_domain_dir = directory / 'open_domain'

    for mc_file in multiple_choice_dir.iterdir():
        answers_path = str(mc_file)
        answers = load_documents(answers_path)
        evaluation_type = 'multiple_choice'
        accuracy = get_accuracy(answers, type=evaluation_type)
        print(f" {answers_path} Accuracy: {accuracy:.2f}")
        
        # Append to log file
        with open("results/accuracy.txt", "a") as log_file:
            log_file.write(f"{answers_path}\tAccuracy: {accuracy:.3f}\n")
    
    for od_file in open_domain_dir.iterdir():
        answers_path = str(od_file)
        answers = load_documents(answers_path)
        evaluation_type = 'binary'
        accuracy = get_accuracy(answers, type=evaluation_type)
        print(f" {answers_path} Accuracy: {accuracy:.2f}")
        
        # Append to log file
        with open("results/accuracy.txt", "a") as log_file:
            log_file.write(f"{answers_path}\tAccuracy: {accuracy:.3f}\n")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
