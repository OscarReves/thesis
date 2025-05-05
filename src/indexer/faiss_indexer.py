import faiss
from src.utils import load_documents
import numpy as np

class FaissIndexer:
    def __init__(self, embedder, index_path):
        self.embedder = embedder  # BERT, mBERT, etc.
        self.index_path = index_path

    def index_documents(self, documents, with_uid=False):
        embeddings = self.embedder.encode(documents)
        faiss.normalize_L2(embeddings)
        # Build and save FAISS index
        dim = self.embedder.model.config.hidden_size
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

    def index_documents_with_uid(self, documents):
        embeddings = self.embedder.encode(documents)
        faiss.normalize_L2(embeddings)
        # Build and save FAISS index
        dim = self.embedder.model.config.hidden_size
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        uids = np.array(documents["uid"], dtype=np.int64)
        index.add_with_ids(embeddings, uids)

        faiss.write_index(index, self.index_path)


    def index_directory(self, document_paths, batch_size):
        paths = [str(p) for p in document_paths]
        dataset = load_documents(paths)  # Should return a Dataset with 'uid' column
        embeddings = self.embedder.encode(dataset,batch_size)
        faiss.normalize_L2(embeddings)

        dim = self.embedder.model.config.hidden_size
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        uids = np.array(dataset["uid"], dtype=np.int64)
        index.add_with_ids(embeddings, uids)

        faiss.write_index(index, self.index_path)