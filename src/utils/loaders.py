from datasets import load_dataset, Dataset, load_from_disk
import json
from pathlib import Path
import os 
import re
from torch.utils.data import random_split
import torch

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

def load_squad(path, prepend_with_title=False, with_context=False, silent=False):
    data = load_documents(path, silent=True)['data'][0]
    records = []
    context_id_counter = 0

    for entry in data:
        article_title = entry.get("title")
        for i, para in enumerate(entry.get("paragraphs", [])):
            context = para.get("context")
            for qa in para.get("qas", []):
                question = qa.get("question")
                if prepend_with_title:
                    question = article_title + " - " + question
                answers = [a["text"] for a in qa.get("answers", [])]
                records.append({
                    "question": question,
                    "answers": answers,
                    **({"context": context, "context_id": context_id_counter} if with_context else {})
                })
            context_id_counter += 1 # if you indent this line incorrectly literally everything breaks
            # pay attention ffs
            # 3 hours wasted 

    if not silent:
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

def load_citizenship_questions(silent=False, split=False):
    dataset = load_dataset("sorenmulli/citizenship-test-da", "default")
    correct_mapping = {'A': 0, 'B': 1, 'C': 2}
    records = []

    # re-formats so format is equivalent to news questions
    for entry in dataset['train']:
        question = entry['question']
        options = [entry['option-A'], entry['option-B'], entry['option-C']]
        correct_idx = correct_mapping[entry['correct']]
        correct = entry['correct']
        answer = options[correct_idx]
        mc_answer = correct + ": " + answer
        id = entry['index']
        records.append({
            "id": id,
            "question": question,
            "options": options,
            "correct_idx": correct_idx,
            "answer": answer,
            "mc_answer": mc_answer
        })
    
    ds = Dataset.from_list(records)

    if split:
        seed: int = 42,
        train_ratio: float = 0.8
        train_test = ds.train_test_split(
            test_size=1 - train_ratio, seed=seed, shuffle=True
        )
        train_ds, test_ds = train_test["train"], train_test["test"]
        if not silent:
            print(
                f"Loaded {len(ds)} questions : train {len(train_ds)}, test {len(test_ds)}"
            )
        return train_ds, test_ds

    if not silent:
        print(f"Loaded {len(records)} questions")
    return Dataset.from_list(records)

def load_mkqa(silent=False):
    mkqa = load_dataset('apple/mkqa', trust_remote_code=True)
    records = []
    for sample in mkqa['train']:
        example_id = sample['example_id']
        question = sample['queries']['da']
        answer = sample['answers']['da'][0]['text']
        records.append({
            'example_id'    : example_id,
            'question'      : question,
            'answer'        : answer
        })

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

def load_news(path, silent=False):
    dataset = load_dataset("json",data_files=path, field=None, split='train')
    dataset = dataset.map(
        lambda x: {k : v for k,v in x['quiz'].items()}
        )
    dataset = dataset.map(
        lambda x: {'answer': x['options'][x['correct_idx']]}
        )
    correct_mapping = ['A', 'B', 'C', 'D']
    dataset = dataset.map(
        lambda x: {'mc_answer': 
                   correct_mapping[x['correct_idx']] + ": " + x['options'][x['correct_idx']]}
        )
    if not silent:
        print(f"{len(dataset)} questions loaded")
    return dataset

def load_retrieval_corpus(max_samples=None):
    dataset = load_dataset('ThatsGroes/synthetic-from-retrieval-tasks-danish')
    def unwrap_and_merge(example, idx):
        try:
            json_str = re.sub(r"^\s*```json\s*\n|\n\s*```\s*$", "", example["response"], flags=re.DOTALL)
            parsed = json.loads(json_str)
        except Exception:
            parsed = {}

        return {
            "uid": idx,  # <-- integer ID for FAISS
            "query": parsed.get("user_query", ""),
            "text": parsed.get("positive_document", ""),
        }

    dataset = dataset["train"].map(
        unwrap_and_merge,
        with_indices=True,
        features=None,
    )

    if max_samples:
        dataset = dataset.select(range(max_samples))

    return dataset

def load_web_faq(path, test=False, max_samples=None):
    dataset = load_from_disk(path)
    if test:
        # Split sizes
        total_samples=len(dataset)
        print(f"Length of dataset before split: {total_samples}")
        train_size = int(0.8 * total_samples)
        val_size = int(0.01 * total_samples)
        test_size = total_samples - (train_size + val_size)
        # # Generate shuffled indices with PyTorch
        # generator = torch.Generator().manual_seed(42)
        # perm = torch.randperm(len(dataset), generator=generator).tolist()

        # # Split indices
        # train_indices = perm[:train_size]
        # test_indices = perm[train_size:]

        generator = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(dataset), generator=generator).tolist()

        train_indices = perm[:train_size]
        val_indices = perm[train_size:train_size + val_size]
        test_indices = perm[train_size + val_size:]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)


        # Create HF subsets using .select()
        train_dataset = dataset.select(train_indices)
        test_dataset = dataset.select(test_indices)
        dataset = test_dataset
        print(f"Length of dataset after split: {len(dataset)}")

    if max_samples:
        dataset = dataset.select(range(max_samples))
    print(f"Loaded {len(dataset)} documents from dataset (TEST = {test})")
    column_renames = {
    "question": "query",
    "answer": "text"
    }
        
    for old, new in column_renames.items():
        dataset = dataset.rename_column(old, new)
    #dataset = dataset.map(lambda example, idx: {"uid": idx}, with_indices=True)
    dataset = dataset.add_column("uid", list(range(len(dataset))))
    return dataset

def load_knowledge_base(path, type, silent=False):
    if type == "squad":
        return load_documents(path, silent)
    if type == "wiki":
        return load_documents_from_directory(path, silent)
    if type == "news":
        return load_documents(path, silent)

def load_questions_by_type(path, type, silent=False, split = False):
    if type == "squad":
        return load_squad(path, silent)
    if type == "squad_with_title":
        return load_squad(path, silent=silent, prepend_with_title=True)
    if type == "squad_rewritten":
        return load_squad_rewritten(path, silent)
    if type == "news":
        return load_news(path, silent)
    if type == "citizenship":
        return load_citizenship_questions(silent, split)
    if type == "mkqa":
        return load_mkqa(silent)
    if type == "custom":
        return load_documents(path)

# == For saving == 

def save_as_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(data)} results to {path}...")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data.to_list(), fp, indent=2, ensure_ascii=False) 
    print(f"{len(data)} results saved to {path}")

def save_to_json(data, path, result_type = "results"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(data)} {result_type} to {path}...")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False) 
    print(f"{len(data)} {result_type} saved to {path}")