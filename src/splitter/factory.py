from .splitters import LangChainSplitter, ParagraphSplitter

SPLITTER_REGISTRY = {
    "langchain" : LangChainSplitter,
    "paragraph" : ParagraphSplitter
}

def get_splitter(name, **kwargs):
    if name not in SPLITTER_REGISTRY:
        raise ValueError(f"Unknown splitter: {name}")
    return SPLITTER_REGISTRY[name](**kwargs)