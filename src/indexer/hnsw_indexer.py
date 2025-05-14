import hnswlib
import numpy as np

class HNSWIndexer:
    def __init__(self, embedder, index_path):
        self.embedder = embedder
        self.index_path = index_path

    def index_documents_with_uid(self, documents, batch_size):
        embeddings = self.embedder.encode(documents, batch_size)
        dim = embeddings.shape[1]

        uids = np.array(documents["uid"], dtype=np.int64)

        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings, uids)
        index.save_index(self.index_path)
