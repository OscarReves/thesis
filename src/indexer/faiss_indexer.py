import faiss
from src.utils import load_documents
import numpy as np
import gc
from tqdm import tqdm
import psutil


class FaissIndexer:
    def __init__(self, embedder, index_path):
        self.embedder = embedder  # BERT, mBERT, etc.
        self.index_path = index_path

    def index_documents(self, documents, batch_size=16):
        # indexes a single HF-dataset 
        embeddings = self.embedder.encode(documents,batch_size)
        faiss.normalize_L2(embeddings)
        # Build and save FAISS index
        dim = self.embedder.model.config.hidden_size
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

    def index_documents_with_uid(self, documents, batch_size):
        # indexes a single HF-dataset with a uid-mapping 
        embeddings = self.embedder.encode(documents,batch_size)
        faiss.normalize_L2(embeddings)
        # Build and save FAISS index
        dim = self.embedder.model.config.hidden_size
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        uids = np.array(documents["uid"], dtype=np.int64)
        index.add_with_ids(embeddings, uids)

        faiss.write_index(index, self.index_path)

    def index_directory(self, document_paths, batch_size):
        # Load the dataset (assumes memory-mapped HF Dataset)
        paths = [str(p) for p in document_paths]
        dataset = load_documents(paths)  # should return a Dataset with 'uid'
        dim = self.embedder.model.config.hidden_size

        # Create a single FAISS index
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        outer_batch_size = batch_size * 100  # Controls how much to embed in one go
        process = psutil.Process()

        for start in tqdm(range(0, len(dataset), outer_batch_size), desc="Indexing"):
            end = start + outer_batch_size
            batch = dataset[start:end]

            # Encode and normalize
            embeddings = self.embedder.encode(batch, batch_size=batch_size)
            faiss.normalize_L2(embeddings)

            # Get UIDs
            uids = np.array(batch["uid"], dtype=np.int64)

            # Add to index
            index.add_with_ids(embeddings, uids)

            mem_gb = process.memory_info().rss / 1e9
            print(f"Added docs {start} to {end} | Memory usage: {mem_gb:.2f} GB")

            del embeddings, uids, batch
            gc.collect()

        # Save the full index
        faiss.write_index(index, self.index_path)
        print(f"Index saved to {self.index_path}")


    def build_index_from_backup_embeddings(
        self,
        data_path='data/wiki/embeddings_backup.npz',
        batch_size=100000
            ):
        index_save_path=self.index_path
        # Load saved data
        print(f"Loading embeddings and uids from: {data_path}")
        data = np.load(data_path)
        embeddings = data["embeddings"]
        uids = data["uids"]

        # Normalize if needed
        faiss.normalize_L2(embeddings)

        # Initialize index
        dim = embeddings.shape[1]
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        # Add in batches
        for start in range(0, len(embeddings), batch_size):
            end = start + batch_size
            index.add_with_ids(embeddings[start:end], uids[start:end])
            print(f"Indexed batch {start} to {end}")

        # Save index
        faiss.write_index(index, index_save_path)
        print(f"FAISS index saved to: {index_save_path}")

class FaissIndexerOld:
    # THIS INDEXER INCORRECTLY USES L2 DISTANCE 
    # SAVED FOR COMPLETENESS
    
    def __init__(self, embedder, index_path):
        self.embedder = embedder  # BERT, mBERT, etc.
        self.index_path = index_path

    def index_documents(self, documents, batch_size=16):
        # indexes a single HF-dataset 
        embeddings = self.embedder.encode(documents,batch_size)
        faiss.normalize_L2(embeddings)
        # Build and save FAISS index
        dim = self.embedder.model.config.hidden_size
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

    def index_documents_with_uid(self, documents, batch_size):
        # indexes a single HF-dataset with a uid-mapping 
        embeddings = self.embedder.encode(documents,batch_size)
        faiss.normalize_L2(embeddings)
        # Build and save FAISS index
        dim = self.embedder.model.config.hidden_size
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        uids = np.array(documents["uid"], dtype=np.int64)
        index.add_with_ids(embeddings, uids)

        faiss.write_index(index, self.index_path)

    def index_directory(self, document_paths, batch_size):
        # Load the dataset (assumes memory-mapped HF Dataset)
        paths = [str(p) for p in document_paths]
        dataset = load_documents(paths)  # should return a Dataset with 'uid'
        dim = self.embedder.model.config.hidden_size

        # Create a single FAISS index
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        outer_batch_size = batch_size * 100  # Controls how much to embed in one go
        process = psutil.Process()

        for start in tqdm(range(0, len(dataset), outer_batch_size), desc="Indexing"):
            end = start + outer_batch_size
            batch = dataset[start:end]

            # Encode and normalize
            embeddings = self.embedder.encode(batch, batch_size=batch_size)
            faiss.normalize_L2(embeddings)

            # Get UIDs
            uids = np.array(batch["uid"], dtype=np.int64)

            # Add to index
            index.add_with_ids(embeddings, uids)

            mem_gb = process.memory_info().rss / 1e9
            print(f"Added docs {start} to {end} | Memory usage: {mem_gb:.2f} GB")

            del embeddings, uids, batch
            gc.collect()

        # Save the full index
        faiss.write_index(index, self.index_path)
        print(f"Index saved to {self.index_path}")


    def build_index_from_backup_embeddings(
        self,
        data_path='data/wiki/embeddings_backup.npz',
        batch_size=100000
            ):
        index_save_path=self.index_path
        # Load saved data
        print(f"Loading embeddings and uids from: {data_path}")
        data = np.load(data_path)
        embeddings = data["embeddings"]
        uids = data["uids"]

        # Normalize if needed
        faiss.normalize_L2(embeddings)

        # Initialize index
        dim = embeddings.shape[1]
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        # Add in batches
        for start in range(0, len(embeddings), batch_size):
            end = start + batch_size
            index.add_with_ids(embeddings[start:end], uids[start:end])
            print(f"Indexed batch {start} to {end}")

        # Save index
        faiss.write_index(index, index_save_path)
        print(f"FAISS index saved to: {index_save_path}")

