import argparse
import yaml
from src.indexer import FaissIndexer
from src.utils import load_wiki_file_paths
from src.embedder import get_embedder
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    docs_path = config["chunked_path"] # load chunked documents
    index_path = config["index_path"]
    embedder_name = config["embedder_name"]
    device = config['device']
    batch_size = config['batch_size']

    print("Loading document paths...")
    document_paths = load_wiki_file_paths(docs_path)
    
    print(f"Initializing embedder ({embedder_name})...", flush=True)
    embedder = get_embedder(name = embedder_name, device = device)
    
    print("Loading indexer...")
    indexer = FaissIndexer(embedder=embedder, index_path=index_path)
    
    print("Indexing documents...")
    indexer.build_index_from_backup_embeddings()

    print(f"Index built and saved to {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
