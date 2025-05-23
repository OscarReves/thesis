import json
from src.utils import batch_iterator
from src.utils import save_to_json

def evaluate_answers(answer_dataset, evaluator, save_path, batch_size=16):
    results = []

    for batch in batch_iterator(answer_dataset, batch_size, 
                                description=f"Evaluating answers in batches of {batch_size}"):
        
        # Now access fields directly (vectorized or individually)
        questions = batch["question"]
        generated_answers = batch["generated_answer"]
        reference_answers = batch["reference_answer"]

        evaluations = evaluator.evaluate_batch(
            questions=questions,
            generated_answers=generated_answers,
            reference_answers=reference_answers
        )

        for sample, evaluation in zip(batch, evaluations):
            results.append({
                "question"         : sample['question'],
                "generated_answer" : sample['generated_answer'],
                "reference_answer" : sample['reference_answer'],
                "evaluation"       : evaluation
            })

    save_to_json(results,path=save_path)
