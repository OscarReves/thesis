from src.utils import chunk_dataset, save_as_json
from transformers import AutoTokenizer
from pathlib import Path
import json
from src.utils import load_wiki_file_paths, load_wiki_articles, chunk_dataset
from tqdm import tqdm
from datasets import disable_progress_bar


def chunk_and_save(documents, tokenizer_name, save_path):
    
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print("Chunking dataset...")
    chunked = chunk_dataset(documents, tokenizer)

    chunked.to_json(save_path) # removed lines=False
    #save_as_json(data=chunked,path=save_path)


def chunk_multiple(dump_dir, out_dir, tokenizer_name):
    # chunks and saves multiple wiki dump files
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    disable_progress_bar()
    file_paths = load_wiki_file_paths(dump_dir)

    for file_path in tqdm(file_paths, desc="Chunking files"):
        rel_path = file_path.relative_to(dump_dir)  # e.g., AA/wiki_03
        out_path = Path(out_dir) / rel_path.with_suffix(".json")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue  # Skip already processed

        documents = load_wiki_articles(str(file_path), silent=True)
        chunked = chunk_dataset(documents, tokenizer)

        chunked.to_json(out_path)

    print(f"{len(file_paths)} saved in {out_dir}")