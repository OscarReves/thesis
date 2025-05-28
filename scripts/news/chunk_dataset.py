import argparse
import yaml
from src.utils import load_raw_articles
from src.pipeline import chunk_and_save
from src.splitter import get_splitter
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    docs_path = config["documents_path"]
    save_path = config['chunked_path']
    splitter_name = config['splitter_name']
    prepend_with_title=config['prepend_with_title']

    print("Loading documents...")
    documents = load_raw_articles(docs_path)
    
    splitter = get_splitter(splitter_name)

    print(f"Chunking documents WITH TITLE={prepend_with_title}...")
    # this should be expanded to inlcude arguments for chunk size etc
    # also, consider not chunking based on the tokenizer 
    chunk_and_save(documents, splitter, save_path, prepend_with_title=prepend_with_title)

    print(f"Chunked documents saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
