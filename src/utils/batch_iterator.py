from tqdm import tqdm
import math

def batch_iterator(dataset, batch_size):
    total_batches = math.ceil(len(dataset) / batch_size)
    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches):
        yield dataset.select(range(i, min(i + batch_size, len(dataset))))
