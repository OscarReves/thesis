def is_correct(sample):
    return sample['evaluation'].split()[0] == '1'

def is_false(sample):
    return sample['evaluation'].split()[0] == '0'

def get_accuracy(dataset):
    # returns the accuracy for an answer dataset scored with binary evaluation
    correct = dataset.filter(is_correct)
    accuracy = len(correct)/len(dataset)
    return accuracy