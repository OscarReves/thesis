import argparse
import yaml
from src.utils import load_raw_articles
from src.pipeline import chunk_and_save
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    docs_path = config["documents_path"]
    tokenizer_name = config["tokenizer_name"]
    save_path = config['chunked_path']

    print("Loading documents...")
    documents = load_raw_articles(docs_path)
    
    print("Chunking documents...")
    # this should be expanded to inlcude arguments for chunk size etc
    # also, consider not chunking based on the tokenizer 
    chunk_and_save(documents, tokenizer_name, save_path)

    print(f"Chunked documents saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
