import json
from tqdm import tqdm
import os 
from src.utils import save_as_json

def test_retrieval(question_dataset, retriever, save_path, batch_size = 16):
    results = []
    for sample in question_dataset:
        question = sample['question']
        context = retriever.retrieve_titles([question])[0]
        result = {
            "question"  :   question,
            "context"   :   context
            } 
        results.append(result)
    
    

    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

def test_retrieval_with_uid(question_dataset, retriever, save_path, max_samples=100):
    print(f"Testing uid based retrieval...")
    results = []
    question_dataset = question_dataset.select(range(max_samples))
    for sample in tqdm(question_dataset):
        question = sample['question']
        context = retriever.retrieve_titles_with_uid([question])
        result = {
            "question"  :   question,
            "context"   :   context
            } 
        results.append(result)
    
    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

def test_batched_retrieval_with_uid(question_dataset, retriever, save_path, batch_size = 16, max_samples=1600):
    print(f"Testing uid based retrieval with batching...")
    results = []
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