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
    generator_name = config['generator_name']

    print("Loading generator...")
    generator = get_generator(generator_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
