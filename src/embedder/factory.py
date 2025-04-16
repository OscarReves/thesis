from .embedders import GPT2Embedder, E5Embedder, BertTinyEmbedder

EMBEDDER_REGISTRY = {
    "gpt2"  : GPT2Embedder,
    "e5"    : E5Embedder,
    "bert-tiny" : BertTinyEmbedder
}

def get_embedder(name, **kwargs):
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return EMBEDDER_REGISTRY[name](**kwargs)