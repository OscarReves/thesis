import yaml
from src.utils import save_to_json, load_documents, get_retrieval_accuracy, load_web_faq
from tqdm import tqdm
import os 
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
from datasets import load_dataset



def main():

    # Build dataloader
    dataset_path = 'data/webfaq_danish'
    if os.path.exists(dataset_path):
        print(f'Dataset found on disk at {dataset_path}')
    else:
        # download danish split
        dataset = load_dataset("PaDaS-Lab/webfaq", "dan", split='default')
        # save to disk
        dataset.save_to_disk(dataset_path)
    
    save_path = 'models/e5_finetuned_epoch7.pt'
    device = torch.device("cuda")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    print(f"Loading state dict from {save_path}")
    state_dict = torch.load(save_path)    
    model.eval()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)


    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / count

    global_step = 0

    # Tokenize queries with progress
    # add loading from disk in the future
    
    passage_inputs_path = 'data/training/passage_inputs.pt'
    query_inputs_path = 'data/training/query_inputs.pt' 
    
    if os.path.exists(query_inputs_path) and os.path.exists(passage_inputs_path):
        print(f"Loading pre-tokenized passages from disk at {passage_inputs_path}")
        passage_inputs = torch.load(passage_inputs_path)
        print(f"Loading pre-tokenized queries from disk at {query_inputs_path}")
        query_inputs = torch.load(query_inputs_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large",use_fast=True)
        # tokenize queries
        queries = [f"query: {q}" for q in dataset["query"]]
        query_inputs = {"input_ids": [], "attention_mask": []}
        for q in tqdm(queries, desc="Tokenizing queries"):
            encoded = tokenizer(q, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            query_inputs["input_ids"].append(encoded["input_ids"])
            query_inputs["attention_mask"].append(encoded["attention_mask"])
        query_inputs["input_ids"] = torch.cat(query_inputs["input_ids"])
        query_inputs["attention_mask"] = torch.cat(query_inputs["attention_mask"])
        torch.save(query_inputs, query_inputs_path)
        # tokenize passages
        passages = [f"passage: {p}" for p in dataset["text"]]
        passage_inputs = {"input_ids": [], "attention_mask": []}
        for p in tqdm(passages, desc="Tokenizing passages"):
            encoded = tokenizer(p, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            passage_inputs["input_ids"].append(encoded["input_ids"])
            passage_inputs["attention_mask"].append(encoded["attention_mask"])
        passage_inputs["input_ids"] = torch.cat(passage_inputs["input_ids"])
        passage_inputs["attention_mask"] = torch.cat(passage_inputs["attention_mask"])
        torch.save(passage_inputs, passage_inputs_path)

    # Zip and load into DataLoader
    tensor_dataset = TensorDataset(
        query_inputs["input_ids"],
        query_inputs["attention_mask"],
        passage_inputs["input_ids"],
        passage_inputs["attention_mask"]
    )

    # Split sizes
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        tensor_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)


    def mean_pooling(last_hidden, mask):
        mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def contrastive_loss(q, p, temperature=0.05):
        q = F.normalize(q, dim=1)
        p = F.normalize(p, dim=1)
        logits = torch.matmul(q, p.T) / temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)


    model.eval()
    pbar = tqdm(test_dataloader, desc=f"Calculating test loss")
    total_loss = 0.0
    total_examples = 0.0
    for q_ids, q_mask, p_ids, p_mask in pbar:
        q_ids, q_mask = q_ids.to(device), q_mask.to(device)
        p_ids, p_mask = p_ids.to(device), p_mask.to(device)


        with torch.inference_mode():
            q_out = model(input_ids=q_ids, attention_mask=q_mask)
            p_out = model(input_ids=p_ids, attention_mask=p_mask)

            q_emb = mean_pooling(q_out.last_hidden_state, q_mask)
            p_emb = mean_pooling(p_out.last_hidden_state, p_mask)

            loss = contrastive_loss(q_emb, p_emb)

        print(f"[Step {global_step}] Test Loss: {loss:.4f}")

        pbar.set_postfix(loss=loss.item())

        total_loss += loss
        total_examples += 1
        avg_loss = total_loss / total_examples
        print(f"[Step {global_step}] Average Test Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
