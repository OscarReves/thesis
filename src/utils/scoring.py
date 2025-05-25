import pandas as pd 
import numpy as np
from src.utils import load_documents
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

# === Question Answering Accuracy ===

def is_correct(sample):
    return sample['evaluation'].split()[0] == '1'

def is_false(sample):
    return sample['evaluation'].split()[0] == '0'

def is_correct_mc(sample):
    chosen_option = sample['generated_answer'][0]
    correct_option = sample['reference_answer'][0]
    return chosen_option == correct_option

def get_accuracy(dataset, type='binary'):
    # returns the accuracy for an answer dataset
    metrics = {
        'binary':   is_correct,
        'multiple_choice': is_correct_mc
    }
    metric = metrics[type]
    correct = 0
    for sample in dataset:
        if metric(sample):
            correct += 1
    
    accuracy = correct/len(dataset)
    return accuracy

# === Retrieval Accuracy === 

# def retrieval_success(sample):
#     return (sample['context_id'] in sample['retrieved_uids'])

# def get_retrieval_accuracy(dataset):
#     # assumes columns retrieved_uids and context_id
#     correct = dataset.filter(retrieval_success)
#     accuracy = len(correct)/len(dataset)
#     return accuracy

# def get_retrieval_accuracy(dataset, k=5):
#     def uid_match(sample, batched=True):
#         return sample['uid'] in sample['retrieved_uids'][:k]
    
#     max_k = len(dataset[0]['retrieved_uids'])
#     if k > max_k:
#         print(f"Warning: Attempting to get accuracy for k = {k} but only {max_k} results have been retrieved")

#     correct = dataset.filter(uid_match)
#     accuracy = len(correct)/len(dataset)
#     return accuracy

def get_retrieval_accuracy(dataset, k=5):
    max_k = len(dataset[0]['retrieved_uids'])
    if k > max_k:
        print(f"Warning: Attempting to get accuracy for k = {k} but only {max_k} results have been retrieved")

    # Extract uid and top-k retrieved uids
    uids = np.array([d['uid'] for d in dataset])
    topk_retrieved = [set(d['retrieved_uids'][:k]) for d in dataset]

    # Vectorized comparison using list comprehension (still faster than .filter)
    correct = np.array([uid in retrieved for uid, retrieved in zip(uids, topk_retrieved)])
    return np.mean(correct)


# === Human Annotation === 

def get_human_votes(path_to_csv='results/citizenship/human_evaluation/human_annotation.csv'):
    # returns an array whose entries correspond to the human evaluation by majority vote
    df = pd.read_csv(path_to_csv, index_col=False, header=0)
    df = df.fillna(1/df.shape[0]) # regard blank answers as average 
    df = df.drop('Tidsstempel', axis=1)
    df = df.applymap(lambda x: int(x[0]) if isinstance(x, str) and x else x)
    array = df.to_numpy()
    return array

def get_human_evals(path_to_csv='results/citizenship/human_evaluation/human_annotation.csv'):
    # returns an array whose entries correspond to the human evaluation by majority vote
    df = pd.read_csv(path_to_csv, index_col=False, header=0)
    df = df.fillna(1/df.shape[0]) # regard blank answers as average 
    df = df.drop('Tidsstempel', axis=1)
    df = df.applymap(lambda x: int(x[0]) if isinstance(x, str) and x else x)
    array = df.to_numpy()
    means = array.mean(axis=0)
    result = means.round().astype(int)
    return result

def get_model_evals(path='results/citizenship/human_evaluation/100_balanced_questions'):
    questions = load_documents(path)
    model_evals = np.array([int(x[0]) if x and x[0].isdigit() else np.nan for x in questions['evaluation']])
    return model_evals

def get_eval_accuracy(
        human_path='results/citizenship/human_evaluation/human_annotation.csv',
        model_path='results/citizenship/human_evaluation/100_balanced_questions'
            ):
    human_eval = get_human_evals(human_path)
    model_eval = get_model_evals(model_path)
    return np.mean(human_eval==model_eval)

def get_eval_metrics(
        human_path='results/citizenship/human_evaluation/human_annotation.csv',
        model_path='results/citizenship/human_evaluation/100_balanced_questions'
    ):
    human_eval = get_human_evals(human_path)
    model_eval = get_model_evals(model_path)

    metrics = {
        'accuracy' : np.mean(human_eval == model_eval),
        'precision' : precision_score(human_eval, model_eval, zero_division=0),
        'recall' : recall_score(human_eval, model_eval, zero_division=0),
        'f1' : f1_score(human_eval, model_eval, zero_division=0)
        }
    return metrics

def get_annotater_agreement(path='results/citizenship/human_evaluation/human_annotation.csv'):
    votes = get_human_votes(path)
    def agreement(p1,p2):
        return np.mean(p1 == p2)
    mean = 0
    for i in range(3):
        for j in range(i,3):
            if i != j:
                print(i,j)
                mean += agreement(votes[i],votes[j])
    return mean/len(votes)

def get_model_agreements(path='results/citizenship/human_evaluation/model_evaluations/'):
    directory = Path(path)
    evals = {'Human Majority': get_human_evals()}
    for file in directory.iterdir():
        file_path = directory / file.name
        print(file_path)
        eval = get_model_evals(str(file_path))
        evals.update({file.name[:-7] : eval}) # remove "-binary" suffix for aesthetic reasons
    def agreement(p1,p2):
        return np.mean(p1 == p2)
    scores = {}
    for model,eval in evals.items():
        agreements = []
        for _,eval2 in evals.items():
            agreements.append(agreement(eval,eval2))
        scores.update({model: agreements})
    return scores