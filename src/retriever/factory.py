from .retrievers import E5Retriever, DummyRetriever, BM25Retriever, SparseBM25Retriever, E5RetrieverGPU, E5RetrieverFinetuned

RETRIEVER_REGISTRY = {
    "e5"    : E5Retriever,
    "e5-gpu" : E5RetrieverGPU,
    "dummy" : DummyRetriever,
    "bm25"  : BM25Retriever,
    "bm25-sparse" : SparseBM25Retriever,
    "e5-finetuned" : E5RetrieverFinetuned,
}

def get_retriever(name, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown retriever: {name}")
    return RETRIEVER_REGISTRY[name](**kwargs)

