from datasets import load_dataset, Dataset
import json
from pathlib import Path
import os 


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

def load_squad(path, prepend_with_title=False, with_context=False):
    data = load_documents(path, silent=True)['data'][0]
    records = []
    context_id_counter = 0

    for entry in data:
        article_title = entry.get("title")
        for i, para in enumerate(entry.get("paragraphs", [])):
            context = para.get("context")
            for qa in para.get("qas", []):
                question = qa.get("question")
                if prepend_with_title == True:
                    question = article_title + " - " + question
                answers = [a["text"] for a in qa.get("answers", [])]
                records.append({
                    "question": question,
                    "answers": answers,
                    **({"context": context, "context_id": context_id_counter} if with_context else {})
                })
            context_id_counter += 1 # if you indent this line incorrectly literally everything breaks
            # pay attention ffs
            # 3 hours waster 

    print(f"Loaded {len(records)} questions")
    return Dataset.from_list(records)

def load_squad_rewritten(path, silent=False):
    dataset = load_dataset("json",data_files=path, field=None, split='train')
    dataset = dataset.rename_column("rewritten_question","question")
    if not silent:
        print(f"{len(dataset)} questions loaded")
    return dataset

def load_squad_as_kb(path, silent=False):
    data = load_documents(path, silent=True)['data'][0]
    records = []
    uid_counter = 0

    for entry in data:
        article_title = entry.get("title")
        for para in entry.get("paragraphs", []):
            context = para.get("context")
            records.append({
                    "id" : article_title,
                    "text": context,
                    "uid" : uid_counter
                })
            uid_counter += 1

    print(f"Loaded {len(records)} questions")
    return Dataset.from_list(records)

# == For loading processed data == 

def load_documents(path, silent=False):
    # for loading chunked documents
    dataset = load_dataset("json", data_files=path, split='train')
    if not silent:
        print(f"{len(dataset)} documents loaded")
    return dataset

def load_documents_from_directory(documents_dir, silent=False):
    # for loading chunked documents
    document_paths = load_wiki_file_paths(documents_dir)
    paths = [str(p) for p in document_paths]
    return load_documents(paths, silent=silent)


def load_questions(path, silent=False):
    dataset = load_dataset("json",data_files=path, field=None, split='train')
    if not silent:
        print(f"{len(dataset)} questions loaded")
    return dataset


# == For saving == 

def save_as_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(data)} results to {path}...")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data.to_list(), fp, indent=2, ensure_ascii=False) 
    print(f"{len(data)} results saved to {path}")

