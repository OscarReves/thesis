import faiss
import numpy as np

index = faiss.read_index("data/wiki/embedded/00")
query = np.random.rand(1, 128).astype("float32")
faiss.normalize_L2(query)
D, I = index.search(query, 5)
print(I)
