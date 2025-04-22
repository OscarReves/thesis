from datasets import load_dataset
import json

def load_raw_articles(path, silent=False):
    # for loading raw articles after scraping
    dataset = load_dataset("json", data_files=path, split='train')
    dataset = dataset.filter(lambda x: x['error'] is None) # filter away articles where scraping failed
    if not silent:
        print(f"{len(dataset)} documents loaded")
    return dataset

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

def save_as_json(data, path):
    print(f"Saving {len(data)} results to {path}")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data.to_list(), fp, indent=2, ensure_ascii=False) 