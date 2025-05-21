from .retrievers import E5Retriever, DummyRetriever, BM25Retriever, SparseBM25Retriever

RETRIEVER_REGISTRY = {
    "e5"    : E5Retriever,
    "dummy" : DummyRetriever,
    "bm25"  : BM25Retriever,
    "bm25-sparse" : SparseBM25Retriever
}

def get_retriever(name, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return RETRIEVER_REGISTRY[name](**kwargs)

