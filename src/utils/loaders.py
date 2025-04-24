from datasets import load_dataset
import json
from pathlib import Path


# == For loading raw data ==

def load_raw_articles(path, silent=False):
    # for loading raw news articles after scraping
    dataset = load_dataset("json", data_files=path, split='train')
    dataset = dataset.filter(lambda x: x['error'] is None) # filter away articles where scraping failed
    if not silent:
        print(f"{len(dataset)} documents loaded")
    return dataset

def load_wiki_articles(path, silent=False):
    # for loading wiki articles before chunking
    dataset = load_dataset("json", data_files=path, split='train')
    dataset = dataset.rename_column('text','body')
    if not silent:
        print(f"{len(dataset)} documents loaded")
    return dataset

def load_wiki_file_paths(dump_dir_path="data/wiki/dump", silent=False):
    dump_dir = Path(dump_dir_path)
    file_paths = sorted(dump_dir.glob("*/wiki_*"))  # e.g., AA/wiki_00, AB/wiki_17

    if not silent:
        print(f"Found {len(file_paths)} wiki files under {dump_dir_path}")

    return file_paths

# == For loading processed data == 

def load_documents(path, silent=False):
    # for loading chunked documents
    dataset = load_dataset("json", data_files=path, split='train')
    if not silent:
        print(f"{len(dataset)} documents loaded")
    return dataset

def load_questions(path, silent=False):
    dataset = load_dataset("json",data_files=path, field=None, split='train')
    if not silent:
        print(f"{len(dataset)} questions loaded")
    return dataset


# == For saving == 

def save_as_json(data, path):
    print(f"Saving {len(data)} results to {path}...")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data.to_list(), fp, indent=2, ensure_ascii=False) 
    print(f"{len(data)} results saved to {path}")

