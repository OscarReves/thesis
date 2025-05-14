import faiss
from src.utils import load_documents
import numpy as np
import gc
import tqdm

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


    # def index_directory(self, document_paths, batch_size):
    #     # faiss indexes all files in document_paths
    #     paths = [str(p) for p in document_paths]
    #     dataset = load_documents(paths)  # Should return a Dataset with 'uid' column
    #     embeddings = self.embedder.encode(dataset,batch_size)
    #     faiss.normalize_L2(embeddings)

    #     dim = self.embedder.model.config.hidden_size
    #     base_index = faiss.IndexFlatL2(dim)
    #     index = faiss.IndexIDMap(base_index)

    #     uids = np.array(dataset["uid"], dtype=np.int64)
    #     index.add_with_ids(embeddings, uids)

    #     faiss.write_index(index, self.index_path)


    # def index_directory(self, document_paths, batch_size):
    #     # Load document paths
    #     paths = [str(p) for p in document_paths]
    #     dataset = load_documents(paths)  # Should return a Dataset with 'uid' column

    #     # Encode all documents
    #     embeddings = self.embedder.encode(dataset, batch_size)
    #     faiss.normalize_L2(embeddings)

    #     # Prepare index
    #     dim = self.embedder.model.config.hidden_size
    #     base_index = faiss.IndexFlatL2(dim)
    #     index = faiss.IndexIDMap(base_index)

    #     # Prepare UIDs
    #     uids = np.array(dataset["uid"], dtype=np.int64)

    #     # Save embeddings in case of crash 
    #     data_save_path = 'data/wiki/embeddings_backup'
    #     np.savez(data_save_path, embeddings=embeddings, uids=uids)

    #     # Free unused objects to save RAM
    #     del dataset
    #     gc.collect()

    #     # Incrementally add in batches
    #     add_batch_size = 100000  # Tune this as needed
    #     for start in range(0, len(embeddings), add_batch_size):
    #         end = start + add_batch_size
    #         index.add_with_ids(embeddings[start:end], uids[start:end])
    #         print(f"Added batch {start} to {end}")

    #     # Optionally save the index
    #     faiss.write_index(index, self.index_path)
    
    def index_directory(self, document_paths, batch_size):
        # indexes every document in document_paths
        # to avoid OOM issues, partial indexes are saved then merged
        index_paths = []

        # Load the dataset (assumes memory-mapped HF Dataset)
        paths = [str(p) for p in document_paths]
        dataset = load_documents(paths)  # should return a Dataset with 'uid'
        dim = self.embedder.model.config.hidden_size

        outer_batch_size = batch_size * 100 # controls how much to hold in memory at once 

        for i, start in enumerate(tqdm(range(0, len(dataset), outer_batch_size))): 
            end = start + outer_batch_size
            batch = dataset[start:end]

            # Encode and normalize
            embeddings = self.embedder.encode(batch, batch_size=batch_size)  # inner model batch size
            faiss.normalize_L2(embeddings)

            # Get UIDs
            uids = np.array(batch["uid"], dtype=np.int64)

            # Add to FAISS
            base_index = faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(base_index)
            index.add_with_ids(embeddings, uids)
            partial_index_path = f"{self.index_path}_part_{i}.index"
            faiss.write_index(index, partial_index_path)
            index_paths.append(partial_index_path)
            print(f"Added docs {start} to {end}")

            del index
            gc.collect()

        # Save index
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)
        faiss.merge_into(index, [faiss.read_index(path) for path in index_paths])
        faiss.write_index(index, self.index_path) # this will likely cause OOM errors
                                                # options are to use sharding or a different index type

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

