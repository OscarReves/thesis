import os
import numpy as np
import faiss

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)

def measure_index_and_embeddings(
        index_path="data/wiki/index/index.faiss", 
        data_path="data/wiki/embeddings_backup.npz"
        ):
    
    print("Measuring file sizes...\n")

    if os.path.exists(index_path):
        index_size = get_file_size_mb(index_path)
        print(f"FAISS index file size: {index_size:.2f} MB")
    else:
        print(f"Index file not found: {index_path}")

    if os.path.exists(data_path):
        data_size = get_file_size_mb(data_path)
        print(f"Embeddings (.npz) file size: {data_size:.2f} MB")

        # Optional: load to check shape
        data = np.load(data_path)
        embeddings = data["embeddings"]
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Memory (RAM) if loaded: {embeddings.nbytes / (1024 ** 2):.2f} MB")
    else:
        print(f"Data file not found: {data_path}")

if __name__ == "__main__":
    measure_index_and_embeddings()
