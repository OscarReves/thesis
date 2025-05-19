from datasets import Dataset
from src.utils import load_documents

def get_incorrect(answers_path):
    # returns only rows which were incorrect 
    answers = load_documents(answers_path)
    
    return answers.filter(lambda example: example['evaluation'] == 0)