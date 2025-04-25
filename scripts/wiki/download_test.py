from transformers import AutoTokenizer, AutoModel

model_name = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModel.from_pretrained(model_name, force_download=True)
