from datasets import load_dataset
import json

def load_documents(path):
    dataset = load_dataset("json", data_files=path, split='train')
    dataset = dataset.filter(lambda x: x['error'] is None) # filter away articles where scraping failed
    return dataset

def load_questions(path):
    dataset = load_dataset("json",data_files=path, field=None, split='train')
    return dataset

def save_as_json(data, path):
    print(f"Saving {len(data)} results to {path}")
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False) 