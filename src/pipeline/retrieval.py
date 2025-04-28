import json
from tqdm import tqdm 

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

def test_batched_retrieval_with_uid(question_dataset, retriever, save_path, batch_size = 16):
    print(f"Testing uid based retrieval with batching...")
    results = []
    for i, sample in enumerate(tqdm(question_dataset)):
        question = sample['question']
        context = retriever.retrieve_titles_with_uid([question])
        result = {
            "question"  :   question,
            "context"   :   context
            } 
        results.append(result)
        if i == 99:
            break
    
    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)