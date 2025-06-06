import argparse
import yaml
from src.utils import load_wiki_file_paths
from src.pipeline import chunk_multiple
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from src.splitter import get_splitter
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def main(config_path):
    # chunks a dir full of wiki files 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dump_path = config["dump_path"] # path to parent directory containing wiki files from dump
    out_path = config["chunked_path"]
    splitter_name = config["splitter_name"]

    splitter = get_splitter(splitter_name)

    # iterate through all wiki files
    print(f"Chunking files in {dump_path} and saving to {out_path}...")
    chunk_multiple(
        dump_dir=dump_path, 
        out_dir=out_path,
        splitter=splitter,  
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
