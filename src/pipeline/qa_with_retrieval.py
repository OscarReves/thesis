import json
from tqdm import tqdm
import time

def test_qa_with_retrieval(question_dataset, retriever, generator, save_path, batch_size = 16, max_samples=100):
    # still needs batching 
    question_dataset=question_dataset.select(range(max_samples))
    results = []
    for sample in tqdm(question_dataset):
        question = sample['question']
        context = retriever.retrieve([question])[0]
        
        # you need to also add a method for MC-answering with neg log likelihood
        answer = generator.generate_answer(question,context)
        
        # this method for retrieving the answer needs to be generalized across datasets somehow
        ref_idx = sample['correct_idx']
        reference_answer = sample['options'][ref_idx]

        result = {
            "question"          :   question,
            "context"           :   context, 
            "generated_answer"  :   answer,
            "reference_answer"  :   reference_answer
            } 
        results.append(result)
    
    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False) 

def test_qa_with_retrieval_wiki(question_dataset, retriever, generator, save_path, 
                                batch_size=16, max_samples=100, silent=True):
    question_dataset = question_dataset.select(range(max_samples))
    results = []
    
    total_retrieval_time = 0.0
    total_generation_time = 0.0
    
    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        
        start_retrieval = time.time()
        contexts = retriever.retrieve_with_uid(questions)
        total_retrieval_time = time.time() - start_retrieval
        
        start_generation = time.time()
        answers = generator.generate_batch(questions, contexts)
        total_generation_time = time.time() - start_generation
        
        reference_answers = batch['answers']

        results.extend([{
            "question"         : q,
            "context"          : c,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, c, a, ra in zip(questions, contexts, answers, reference_answers)])
    
        if not silent:
            print(f"\nRetrieval time: {total_retrieval_time:.2f}s")
            print(f"Generation time: {total_generation_time:.2f}s")


    print(f"\nSaving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
