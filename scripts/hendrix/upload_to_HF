import torch
from transformers import AutoModel, AutoTokenizer

def main():
    
    # Load model
    save_path = 'models/e5_finetuned_epoch7.pt'
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    print(f"Loading state dict from {save_path}")
    state_dict = torch.load(save_path)    
    model.load_state_dict(state_dict)

    # Upload to the hub
    
    from huggingface_hub import HfApi, HfFolder, Repository
    model.push_to_hub("E5_finetuned_epoch7")
    tokenizer.push_to_hub("E5_finetuned_epoch7")

if __name__ == "__main__":
    main()
