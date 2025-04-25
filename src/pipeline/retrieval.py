import json

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

def test_retrieval_with_uid(question_dataset, retriever, save_path, batch_size = 16):
    results = []
    for sample in question_dataset:
        question = sample['question']
        context = retriever.retrieve_single_uid([question])
        #context = "TEST_VALUE"
        result = {
            "question"  :   question,
            "context"   :   context
            } 
        results.append(result)
    
    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

def test_retrieve_single_uid(retriever):
    retriever.uid_sanity_test()