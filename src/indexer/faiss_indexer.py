class FaissIndexer:
    def __init__(self, embedder, index_path):
        self.embedder = embedder  # BERT, mBERT, etc.
        self.index_path = index_path

    def index_documents(self, documents):
        embeddings = self.embedder.encode(documents)
        # Build and save FAISS index
        import faiss
        dim = self.embedder.model.config.hidden_size
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
