from datasets import Dataset
from src.utils import load_documents

def get_incorrect(answers_path):
    # returns only rows which were incorrect 
    answers = load_documents(answers_path)
    answers = answers.map(lambda example, idx: {"id": idx}, with_indices=True)
    return answers.filter(lambda example: example['evaluation'][0] == "0")

def get_incorrect_mc(answers_path):
    # returns only rows which were incorrect 
    answers = load_documents(answers_path)
    answers = answers.map(lambda example, idx: {"id": idx}, with_indices=True)
    return answers.filter(lambda example: example['generated_answer'][0] != example['reference_answer'][0])


def get_correct(answers_path):
    # returns only rows which were correct 
    answers = load_documents(answers_path)
    answers = answers.map(lambda example, idx: {"id": idx}, with_indices=True)
    return answers.filter(lambda example: example['evaluation'][0] == "1")