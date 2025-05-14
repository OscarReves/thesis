from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Choose your model (e.g., BERT base)
model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
save_directory = "models/nous-hermes"  # or any path you want

# Download and save tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
