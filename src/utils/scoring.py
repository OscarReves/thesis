import pandas as pd 
import numpy as np
from src.utils import load_documents
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import evaluate

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

# def get_retrieval_accuracy(dataset, k=5):
#     max_k = len(dataset[0]['retrieved_uids'])
#     if k > max_k:
#         print(f"Warning: Attempting to get accuracy for k = {k} but only {max_k} results have been retrieved")

#     # Extract uid and top-k retrieved uids
#     uids = np.array([d['uid'] for d in dataset])
#     topk_retrieved = [set(d['retrieved_uids'][:k]) for d in dataset]

#     # Vectorized comparison using list comprehension (still faster than .filter)
#     correct = np.array([uid in retrieved for uid, retrieved in zip(uids, topk_retrieved)])
#     return np.mean(correct)

def get_retrieval_accuracy(dataset, k=5):
    max_k = len(dataset[0]['retrieved_uids'])
    if k > max_k:
        print(f"Warning: Attempting to get accuracy for k = {k} but only {max_k} results have been retrieved")

    # Extract uid and top-k retrieved uids
    uids = np.array(dataset['uid'])
    retrieved_uids = np.array(dataset['retrieved_uids'])[:, :k]


    accuracy = (uids[:, None] == retrieved_uids).any(axis=1).mean()
    return accuracy

# === Metrics ===

def get_EM(dataset):
    EM = evaluate.load("exact_match")
    generated = dataset['generated_answer']
    ref = dataset['reference_answer']
    results = EM.compute(predictions = generated, references = ref, ignore_case = True, ignore_punctuation=True)
    return results['exact_match']

def get_BERTscore(dataset):
    bertscore = evaluate.load("bertscore")
    generated = dataset['generated_answer']
    ref = dataset['reference_answer']
    results = bertscore.compute(predictions = generated, references = ref, lang="da")
    return results['f1']


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

def get_model_agreements(path='results/citizenship/human_evaluation/model_evaluations/', 
                         include_human=True,
                         file_suffix=''):
    directory = Path(path)
    if include_human:
        evals = {'Human Majority': get_human_evals()}
    else:
        evals = {}
    for file in directory.iterdir():
        if file_suffix in file.name:
            file_path = directory / file.name
            print(file_path)
            eval = get_model_evals(str(file_path))
            evals.update({file.name.replace(file_suffix,'') : eval}) # remove "-binary" suffix for aesthetic reasons
    def agreement(p1,p2):
        return np.mean(p1 == p2)
    scores = {}
    for model,eval in evals.items():
        agreements = []
        for _,eval2 in evals.items():
            agreements.append(agreement(eval,eval2))
        scores.update({model: agreements})
    return scores

def get_easy_hard(path='results/citizenship/test_qa/open_domain_evaluation/',
                  file_suffix=''):
    dir = Path(path)
    evals = []
    first = True
    for file in dir.iterdir():
        if file_suffix in file.name:        
            file_path = dir / file.name
            if first:
                questions = load_documents(str(file_path))
                first=False
            #print(file_path)
            eval = get_model_evals(str(file_path))
            evals.append(eval)
    evals = np.array(evals)
    evals[np.isnan(evals)] = 1/evals.shape[0] # replace nan
    means = evals.mean(axis=0)
    means = evals.mean(axis=0)
    easiest = np.argsort(means)[-5:][::-1]
    hardest = np.argsort(means)[:5][::-1]
    easiest_questions = questions[easiest]['question']
    easiest_scores = means[easiest]
    hardest_questions = questions[hardest]['question']
    hardest_scores = means[hardest]

    res = {
        'easiest' : (easiest_questions,easiest_scores),
        'hardest' : (hardest_questions,hardest_scores)
    }

    return res

def get_most_improved(path='results/citizenship/test_qa/open_domain_evaluation/'):
    dir = Path(path)
    with_context = []
    no_context = []
    first = True    
    for file in dir.iterdir():
        file_path = dir / file.name
        if first:
            questions = load_documents(str(file_path))
            first=False
        #print(file_path)
        eval = get_model_evals(str(file_path))
        if 'with_context' in file.name:        
            with_context.append(eval)
        if 'no_context' in file.name:
            no_context.append(eval)            
    with_context = np.array(with_context)
    no_context = np.array(no_context)
    with_context[np.isnan(with_context)] = 1/with_context.shape[0] # replace nan
    means_with_context = with_context.mean(axis=0)
    no_context[np.isnan(no_context)] = 1/no_context.shape[0] # replace nan
    means_no_context = no_context.mean(axis=0)
    
    means_diff = means_with_context - means_no_context
    
    
    
    easiest = np.argsort(means_diff)[-5:][::-1]
    hardest = np.argsort(means_diff)[:5][::-1]
    easiest_questions = questions[easiest]['question']
    easiest_scores = means_diff[easiest]
    hardest_questions = questions[hardest]['question']
    hardest_scores = means_diff[hardest]

    res = {
        'easiest' : (easiest_questions,easiest_scores),
        'hardest' : (hardest_questions,hardest_scores)
    }

    return res