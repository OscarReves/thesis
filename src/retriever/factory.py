from .retrievers import E5Retriever, DummyRetriever

RETRIEVER_REGISTRY = {
    "e5"    : E5Retriever,
    "dummy" : DummyRetriever
}

def get_retriever(name, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return RETRIEVER_REGISTRY[name](**kwargs)

# import importlib

# def get_retriever(name, **kwargs):
#     module = importlib.import_module(".retrievers", __package__)
#     class_map = {
#         "gpt2": "GPT2Retriever",
#         "e5": "E5Retriever",
#         "bert-tiny": "BertTinyRetriever",
#         "dummy": "DummyRetriever",
#     }
#     if name not in class_map:
#         raise ValueError(f"Unknown retriever: {name}")
#     retriever_class = getattr(module, class_map[name])
#     return retriever_class(**kwargs)
