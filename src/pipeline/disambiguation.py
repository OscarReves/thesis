import json
from src.utils import batch_iterator
import os

def rewrite_questions(question_dataset, generator, save_path, batch_size=16, max_samples=None):
    results = []

    for batch in batch_iterator(question_dataset, batch_size, 
                                description=f"Rewriting questions in batches of {batch_size}",
                                max_samples=max_samples):
        
        # Now access fields directly (vectorized or individually)
        questions = batch["question"]
        contexts = batch["context"]

        rewritten_questions = generator.rewrite_questions(
            questions=questions,
            contexts=contexts
        )

        for sample, rewritten_questions in zip(batch, rewritten_questions):
            results.append({
                "question"  : sample['question'],
                "answers"   : sample['answers']
            })

    # you should make a util function for saving 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"{len(results)} rewritten questions saved to {save_path}")
