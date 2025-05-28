from .embedders import GPT2Embedder, E5Embedder, BertTinyEmbedder, E5EmbedderLocal, E5Finetuned

EMBEDDER_REGISTRY = {
    "gpt2"  : GPT2Embedder,
    "e5"    : E5Embedder,
    "bert-tiny" : BertTinyEmbedder,
    "e5-local" : E5EmbedderLocal,
    "e5-finetuned" : E5Finetuned
}

def get_embedder(name, **kwargs):
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return EMBEDDER_REGISTRY[name](**kwargs)