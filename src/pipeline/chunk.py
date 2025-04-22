from src.utils import chunk_dataset, save_as_json
from transformers import AutoTokenizer

def chunk_and_save(documents, tokenizer_name, save_path):
    
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print("Chunking dataset...")
    chunked = chunk_dataset(documents, tokenizer)

    chunked.to_json(save_path) # removed lines=False
    #save_as_json(data=chunked,path=save_path)