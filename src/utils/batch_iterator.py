from tqdm import tqdm
import math

def batch_iterator(dataset, batch_size, description=None):
    # Load an HF dataset in batches
    # This is apparently unnecessary, you can simply use slice indexing and avoid out of bounds errors
    total_batches = math.ceil(len(dataset) / batch_size)
    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc=description):
        yield dataset.select(range(i, min(i + batch_size, len(dataset))))
