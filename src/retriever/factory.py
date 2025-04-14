from .retrievers import GPT2Retriever, E5Retriever

RETRIEVER_REGISTRY = {
    "gpt2"  : GPT2Retriever,
    "e5"    : E5Retriever,
}

def get_retriever(name, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return RETRIEVER_REGISTRY[name](**kwargs)