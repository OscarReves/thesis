from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

print(f"Downloading model: {model_name}")
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
print("Download complete.")
