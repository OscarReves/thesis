import json
from tqdm import tqdm

def evaluate_answers(answer_dataset, evaluator, save_path, batch_size=16):

    results = []
    for sample in tqdm(answer_dataset):
        question = sample['question']
        generated_answer = sample['generated_answer']
        reference_answer = sample['reference_answer']

        # you need to also add a method for MC-answering with neg log likelihood
        evaluation = evaluator.evaluate_answer(
            question=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer
        )
        
        result = {
            "question"          :   question,
            "generated_answer"  :   generated_answer,
            "reference_answer"  :   reference_answer,
            "evaluation"        :   evaluation
            } 
        results.append(result)
    
    print(f"Saving {len(results)} results to {save_path}")
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False) 