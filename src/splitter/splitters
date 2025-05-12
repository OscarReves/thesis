from langchain.text_splitter import RecursiveCharacterTextSplitter

class LangChainSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        defaults = {
            "chunk_size": 256,
            "chunk_overlap": 128,
            "separators": ["\n\n", "\n", ".", " ", ""],
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
