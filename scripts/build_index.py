import argparse
import yaml
from src.indexer import FaissIndexer
from src.utils import load_documents  
from src.embedder import get_embedder
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    docs_path = config["documents_path"]
    index_path = config["index_path"]
    embedder_name = config["embedder_name"]
    device = config['device']

    documents = load_documents(docs_path)
    embedder = get_embedder(name = embedder_name, device = device)
    indexer = FaissIndexer(embedder=embedder, index_path=index_path)
    indexer.index_documents(documents)

    print(f"Index built and saved to {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
