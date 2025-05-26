import torch
from datasets import load_dataset, Dataset
import argparse

def main():
    dataset_path = '/dtu/p1/oscrev/webfaq_danish'
    pt_path = 'data/training/tokenized_e5_inputs.pt'
    output_path = '/dtu/p1/oscrev/webfaq_danish_pretokenized'
    # Load original dataset
    dataset = load_dataset(dataset_path, split="train")  # adjust split if needed

    # Load pre-tokenized tensors
    tokenized = torch.load(pt_path)
    input_ids = tokenized["input_ids"].tolist()
    attention_mask = tokenized["attention_mask"].tolist()

    # Add tokenized columns to dataset
    dataset = dataset.add_column("input_ids", input_ids)
    dataset = dataset.add_column("attention_mask", attention_mask)

    # Save to disk
    dataset.save_to_disk(output_path)
    print(f"Saved updated dataset to {output_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", required=True, help="Path to original dataset (e.g. ./my_dataset)")
    # parser.add_argument("--pt", required=True, help="Path to .pt file with tokenized inputs")
    # parser.add_argument("--output", required=True, help="Path to save the new dataset")
    # args = parser.parse_args()

    # main(args.dataset, args.pt, args.output)
    main()