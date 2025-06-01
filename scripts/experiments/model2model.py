import yaml
import argparse
from src.utils import load_documents, load_knowledge_base
from src.embedder import get_embedder
from src.retriever import get_retriever
from src.evaluator import get_evaluator
from src.generator import get_generator
from src.pipeline import evaluate_answers, test_qa_with_retrieval
from pathlib import Path
import torch

def main():
    # evaluates every open-domain answer in a directory
    kb = load_knowledge_base(
        path = '/dtu/p1/oscrev/wiki/chunked_paragraph',
        type = 'wiki'
        )
    retriever = get_retriever(
        name='e5',
        documents = kb,
        index_path = '/dtu/p1/oscrev/wiki/index/index_paragraph',
        device = 'cuda',
        top_k = 5
        )
    generator_names = [
        'gemma-9b',
        'nous-hermes-mistral',
        'suzume-llama3',
        'yi-34b',
        'snakmodel'
    ]
    evaluator_names = [
        'nous-hermes-mistral-binary',
        'gemma-9b-binary',
        "suzume-llama3-binary",
        "yi-34b-binary",
        "snakmodel-binary",
    ]
    batch_size = 8

    # iterate through directory 
    questions_path = 'results/citizenship/human_evaluation/100_balanced_questions'
    generator = get_generator(generator_name)
    questions = load_documents(questions_path)
    questions = questions.rename_columns({
        "reference_answer": "answer",
    })

    for generator_name in generator_names:


        save_path = Path('results/citizenship/model_evaluation/answers') / generator_name

        if generator_name == 'yi-34b':
            batch_size = 8
        else:
            batch_size = 32

        test_qa_with_retrieval(
            question_dataset=questions,
            retriever=retriever,
            generator=generator,
            save_path=save_path,
            max_samples=1,
            batch_size=batch_size
        )

        # Free memory
        del generator
        torch.cuda.empty_cache()
        if torch.backends.cuda.is_built():
            torch.cuda.ipc_collect()

    # for evaluator_name in evaluator_names:
    #     evaluator = get_evaluator(evaluator_name)
    #     answers = load_documents(answers_path)

    #     save_path = Path('results/citizenship/model_evaluation/model_evaluations') / evaluator_name

    #     evaluate_answers(
    #         answer_dataset = answers,
    #         evaluator = evaluator,
    #         save_path = save_path,
    #         batch_size=batch_size
    #         )

    #     # Free memory
    #     del evaluator
    #     torch.cuda.empty_cache()
    #     if torch.backends.cuda.is_built():
    #         torch.cuda.ipc_collect()
    
if __name__ == "__main__":
    main()
