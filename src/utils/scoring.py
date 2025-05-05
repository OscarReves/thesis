# === Question Answering Accuracy ===

def is_correct(sample):
    return sample['evaluation'].split()[0] == '1'

def is_false(sample):
    return sample['evaluation'].split()[0] == '0'

def get_accuracy(dataset):
    # returns the accuracy for an answer dataset scored with binary evaluation
    correct = dataset.filter(is_correct)
    accuracy = len(correct)/len(dataset)
    return accuracy

# === Retrieval Accuracy === 

def retrieval_success(sample):
    print(type(sample['context_id']), type(sample['retrieved_uids'][0]))
    return (sample['context_id'] in sample['retrieved_uids'])

def get_retrieval_accuracy(dataset):
    # assumes columns retrieved_uids and context_id
    correct = dataset.filter(retrieval_success)
    accuracy = len(correct)/len(dataset)
    return accuracy