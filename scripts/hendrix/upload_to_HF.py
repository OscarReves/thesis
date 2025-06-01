import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login, whoami
from dotenv import load_dotenv
import os

def main():
    
    # Load model
    save_path = 'models/e5_finetuned_epoch2.pt'
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    print(f"Loading state dict from {save_path}")
    checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict) # is it possible that this didn't work? Because of module?
    
    # Log in to the hub
    load_dotenv()
    token_env_name = 'HUGGINGFACE_TOKEN'
    token = os.getenv(token_env_name)
    if not token:
        raise RuntimeError(f"{token_env_name} not found in environment")
    
    login(token=token)
    print(whoami())

    # save before uploading. Was this the error the whole time?
    model.save_pretrained("models/e5_finetuned_epoch2")

    # Upload to the hub
    model.push_to_hub("e5_finetuned_epoch2")
    tokenizer.push_to_hub("e5_finetuned_epoch2")

if __name__ == "__main__":
    main()
