import yaml
import argparse
from src.utils import load_documents
from src.evaluator import get_evaluator
from src.pipeline import evaluate_answers
from pathlib import Path
import torch

def main():
    # evaluates every open-domain answer in a directory
    evaluator_names = [
        'nous-hermes-mistral-binary',
        'gemma-9b-binary',
        "suzume-llama3-binary",
        "yi-34b-binary",
        "snakmodel-binary",
    ]
    batch_size = 8

    # iterate through directory 
    answers_path = 'results/citizenship/human_evaluation/100_balanced_questions'

    for evaluator_name in evaluator_names:
        evaluator = get_evaluator(evaluator_name)
        answers = load_documents(answers_path)

        save_path = Path('results/citizenship/human_evaluation/model_evaluations') / evaluator_name

        evaluate_answers(
            answer_dataset = answers,
            evaluator = evaluator,
            save_path = save_path,
            batch_size=batch_size
            )

        # Free memory
        del evaluator
        torch.cuda.empty_cache()
        if torch.backends.cuda.is_built():
            torch.cuda.ipc_collect()
    
if __name__ == "__main__":
    main()
