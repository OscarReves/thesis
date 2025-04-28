import json
def test_qa_with_retrieval(question_dataset, retriever, generator, save_path, batch_size = 16, max_samples=100):
    # still needs batching 
    question_dataset=question_dataset.select(range(max_samples))
    results = []
    for sample in question_dataset:
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

def test_qa_with_retrieval_wiki(question_dataset, retriever, generator, save_path, batch_size = 16, max_samples=100):
    # still needs batching 
    question_dataset=question_dataset.select(range(max_samples))
    results = []
    for sample in question_dataset:
        question = sample['question']
        context = retriever.retrieve([question])[0]
        
        answer = generator.generate_answer(question,context)
        
        # this method for retrieving the answer needs to be generalized across datasets somehow
        reference_answer = sample['answers'][0] # currently just looks at first answer

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