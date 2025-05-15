# === Question Answering Accuracy ===

def is_correct(sample):
    return sample['evaluation'].split()[0] == '1'

def is_false(sample):
    return sample['evaluation'].split()[0] == '0'

def is_correct_mc(sample):
    chosen_option = sample['generated_answer'][0]
    correct_option = sample['reference_answer'][0]
    return chosen_option == correct_option

# def get_accuracy(dataset, type='binary'):
#     # returns the accuracy for an answer dataset
#     metrics = {
#         'binary':   is_correct,
#         'multiple_choice': is_correct_mc
#     }
#     metric = metrics[type]
#     correct = dataset.filter(metric, batched=False)
#     accuracy = len(correct)/len(dataset)
#     return accuracy

def get_accuracy(dataset, type='binary'):
    # returns the accuracy for an answer dataset
    metrics = {
        'binary':   is_correct,
        'multiple_choice': is_correct_mc
    }
    metric = metrics[type]
    correct = 0
    for sample in dataset:
        if is_correct(sample):
            correct += 1
    
    accuracy = correct/len(dataset)
    return accuracy



# === Retrieval Accuracy === 

def retrieval_success(sample):
    return (sample['context_id'] in sample['retrieved_uids'])

def get_retrieval_accuracy(dataset):
    # assumes columns retrieved_uids and context_id
    correct = dataset.filter(retrieval_success)
    accuracy = len(correct)/len(dataset)
    return accuracy