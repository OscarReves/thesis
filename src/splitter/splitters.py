from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

class LangChainSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        defaults = {
            "chunk_size": 256,
            "chunk_overlap": 128,
            "separators": ["\n\n", "\n", ".", " ", ""],
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

class ParagraphSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        defaults = {
            "chunk_size": 256, # if you don't pass a tokenizer this referes to character count, not word count 
            "chunk_overlap": 128,
            "separators": ["\n\n", "\n"],
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

class WordCountTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        defaults = {
            "chunk_size": 128,
            "chunk_overlap": 56,
            "separators": ["\n\n", "\n", " ", ""],  # fallback enabled
        }
        defaults.update(kwargs)
        super().__init__(length_function=lambda text: len(text.split()), **defaults)


class TokenizedSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, min_chunk_tokens=100, tokenizer_name="bert-base-uncased", **kwargs):
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        defaults = {
            "chunk_size": 256,
            "chunk_overlap": 128,
            "separators": ["\n\n", "\n"],
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def num_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def merge_small_chunks(self, chunks):
        merged = []
        buffer = ""

        for chunk in chunks:
            combined = (buffer + " " + chunk).strip() if buffer else chunk
            if self.num_tokens(combined) < self.min_chunk_tokens:
                buffer = combined
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append(chunk)

        if buffer:
            merged.append(buffer)

        return merged

    def split_text(self, text):
        chunks = super().split_text(text)
        return self.merge_small_chunks(chunks)
