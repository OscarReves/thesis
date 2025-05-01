from tqdm import tqdm
import math

def batch_iterator(dataset, batch_size, description=None, max_samples=None):
    # Load an HF dataset in batches
    # This is apparently unnecessary, you can simply use slice indexing and avoid out of bounds errors
    if not max_samples:
        max_samples=len(dataset)
    total_batches = math.ceil(max_samples / batch_size)
    for i in tqdm(range(0, max_samples, batch_size), total=total_batches, desc=description):
        yield dataset.select(range(i, min(i + batch_size, len(dataset))))
