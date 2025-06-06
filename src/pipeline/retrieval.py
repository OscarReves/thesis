import json
from tqdm import tqdm
import os 
from src.utils import save_to_json

def test_retrieval(question_dataset, retriever, save_path, batch_size = 16, max_samples=None):
    print(f"Testing retrieval...")
    results = []
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    for sample in question_dataset:
        question = sample['question']
        context = retriever.retrieve([question])[0]
        result = {
            "question"  :   question,
            "context"   :   context
            } 
        results.append(result)
    
    save_to_json(results,save_path)

def test_retrieval_with_uid(question_dataset, retriever, save_path, max_samples=None, batch_size=16):
    print(f"Testing uid based retrieval...")
    results = []
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    for sample in tqdm(question_dataset):
        question = sample['question']
        context_id = sample['context_id']
        retrieved_uids = retriever.retrieve_uids([question])
        result = {
            "question"  :   question,
            "context_id"   :  context_id,
            "retrieved_uids" : retrieved_uids[0]
            } 
        results.append(result)
    
    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

def test_batched_retrieval_with_uid(question_dataset, retriever, save_path, batch_size = 16, max_samples=None):
    print(f"Testing uid based retrieval with batching...")
    results = []
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    for i in (tqdm(range(0, len(question_dataset), batch_size), desc=f"Retrieving context in batches of {batch_size}")):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        context_ids = batch['context_id']
        retrieved_uids = retriever.retrieve_uids(questions)
        results.extend([
            {"question": q, "context_id": c, "retrieved_uids": r}
            for q,c,r in zip(questions,context_ids,retrieved_uids)
        ])

    path = save_path
    data = results
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(data)} results to {path}...")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False) 
    print(f"{len(data)} results saved to {path}")

def batched_retrieval_with_uid(question_dataset, retriever, save_path, batch_size = 16, max_samples=None):
    print(f"Testing uid based retrieval with batching...")
    results = []
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    for i in (tqdm(range(0, len(question_dataset), batch_size), desc=f"Retrieving context in batches of {batch_size}")):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        contexts = retriever.retrieve_titles_with_uid(questions)
        results.extend([
            {"question": q, "contexts": c}
            for q,c in zip(questions,contexts)
        ])

    path = save_path
    data = results
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(data)} results to {path}...")
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False) 
    print(f"{len(data)} results saved to {path}")