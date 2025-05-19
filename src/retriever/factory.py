from .retrievers import E5Retriever, DummyRetriever

RETRIEVER_REGISTRY = {
    "e5"    : E5Retriever,
    "dummy" : DummyRetriever
}

def get_retriever(name, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return RETRIEVER_REGISTRY[name](**kwargs)

