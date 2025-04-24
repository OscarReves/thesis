from datasets import Dataset

def chunk_text(text, tokenizer, chunk_size=256, stride=32):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = [
        tokens[i:i+chunk_size]
        for i in range(0, len(tokens), chunk_size - stride)
    ]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def chunk_sample(batch, tokenizer):
    all_ids, all_texts = [], []
    for title, body in zip(batch['title'], batch['body']):
        chunks = chunk_text(body, tokenizer)
        ids = [f"{title}_{i}" for i in range(len(chunks))]
        all_ids.extend(ids)
        all_texts.extend(chunks)
    return {"id": all_ids, "text": all_texts}

def chunk_dataset(dataset, tokenizer):
    # chunks a dataset and returns as a dataset
    chunked = dataset.map(
    chunk_sample,
    fn_kwargs= {"tokenizer" : tokenizer},
    remove_columns=dataset.column_names,
    batched=True,
    )
    return chunked

