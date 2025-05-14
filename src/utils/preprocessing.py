from datasets import Dataset
from src.utils import load_squad_as_kb

def chunk_text(text, tokenizer, chunk_size=256, stride=32):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokens[i:i+chunk_size]
        for i in range(0, len(tokens), chunk_size - stride)
    ]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def chunk_text_fixed_tokens(text, chunk_size=256, stride = 128):
    tokens = text.split()
    chunks = [
        " ".join(tokens[i:i + chunk_size])
        for i in range(0, len(tokens), chunk_size - stride)
    ]
    return chunks

# def chunk_sample(batch, tokenizer):
#     all_ids, all_texts = [], []
#     for title, body in zip(batch['title'], batch['body']):
#         chunks = chunk_text(body, tokenizer)
#         ids = [f"{title}_{i}" for i in range(len(chunks))]
#         all_ids.extend(ids)
#         all_texts.extend(chunks)
#     return {"id": all_ids, "text": all_texts}

# def chunk_sample_fixed(batch):
#     all_ids, all_texts = [], []
#     for title, body in zip(batch['title'], batch['body']):
#         chunks = chunk_text_fixed_tokens(body)
#         ids = [f"{title}_{i}" for i in range(len(chunks))]
#         all_ids.extend(ids)
#         all_texts.extend(chunks)
#     return {"id": all_ids, "text": all_texts}

# def chunk_sample_langchain(batch):
#     all_ids, all_texts = [], []
#     for title, body in zip(batch['title'], batch['body']):
#         chunks = splitter.split_text(body)
#         ids = [f"{title}_{i}" for i in range(len(chunks))]
#         all_ids.extend(ids)
#         all_texts.extend(chunks)
#     return {"id": all_ids, "text": all_texts}

def chunk_sample(batch, splitter, prepend_with_title=True):
    all_ids, all_texts = [], []
    for title, body in zip(batch['title'], batch['body']):
        chunks = splitter.split_text(body)
        ids = [f"{title}_{i}" for i in range(len(chunks))]
        if prepend_with_title:
            chunks = [f"Title: {title}\n Text: {chunk}" for chunk in chunks]
        all_ids.extend(ids)
        all_texts.extend(chunks)
    return {"id": all_ids, "text": all_texts}


# def chunk_dataset(dataset, tokenizer):
#     # chunks a dataset and returns as a dataset
#     if tokenizer:
#         chunked = dataset.map(
#             chunk_sample,
#             fn_kwargs= {"tokenizer" : tokenizer},
#             remove_columns=dataset.column_names,
#             batched=True,
#         )
#     else:
#         chunked = dataset.map(
#             chunk_sample_fixed,
#             remove_columns=dataset.column_names,
#             batched=True,
#         )
#     return chunked

def chunk_dataset(dataset, splitter):
    chunked = dataset.map(
        chunk_sample,
        fn_kwargs= {"splitter" : splitter},
        remove_columns=dataset.column_names,
        batched=True
    )

    return chunked

def save_squad_contexts(load_path, save_path):
    # saves contexts from SQuAD so they are ready to be indexed
    dataset = load_squad_as_kb(load_path)
    dataset.to_json(save_path)

    print(f"{len(dataset)} contexts saved in {save_path}")