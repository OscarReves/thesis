from src.utils import chunk_dataset, save_as_json
from transformers import AutoTokenizer
from pathlib import Path
import json
from src.utils import load_wiki_file_paths, load_wiki_articles, chunk_dataset
from tqdm import tqdm
from datasets import disable_progress_bar



def chunk_and_save(documents, splitter, save_path, prepend_with_title=True):
    
    print("Chunking dataset...")
    chunked = chunk_dataset(documents, splitter, prepend_with_title=prepend_with_title)

    #chunked.to_json(save_path) # removed lines=False
    save_as_json(data=chunked,path=save_path)

def chunk_multiple(dump_dir, out_dir, splitter, test=False):
    # chunks and saves multiple wiki dump files
    # you should absolutely parallelize this 
    
    disable_progress_bar()
    file_paths = load_wiki_file_paths(dump_dir)

    uid_counter = 0

    for file_path in tqdm(file_paths, desc="Chunking files"):
        # chunks each file in file_path
        rel_path = file_path.relative_to(dump_dir)  # e.g., AA/wiki_03
        out_path = Path(out_dir) / rel_path.with_suffix(".jsonl")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Chunk file 
        documents = load_wiki_articles(str(file_path), silent=True)
        chunked = chunk_dataset(documents, splitter)

        # Assign incremental integer UIDs
        num_chunks = len(chunked)
        uids = list(range(uid_counter, uid_counter + num_chunks))
        chunked = chunked.add_column("uid", uids)
        uid_counter += num_chunks
        
        # Save
        chunked.to_json(out_path)
        if test:
            break

    print(f"{len(file_paths)} files saved in {out_dir}")

    