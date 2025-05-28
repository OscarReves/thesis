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
import numpy as np


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

    dataset = load_web_faq(dataset_path)
    device = torch.device("cuda")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    model.train()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler(device='cuda')  # for mixed precision

    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / count


    # Logger 
    writer = SummaryWriter(log_dir="./logs/e5_finetune")

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

    if True:
        print("DATASET LIMITED TO 20K FOR DRY RUN")
        print("CHANGE BOOL TO RUN PROPERLY")
        tensor_dataset = tensor_dataset[:20000] # for dry run


    # Split sizes
    train_size = int(0.8 * len(tensor_dataset))
    val_size = int(0.01 * len(tensor_dataset))
    test_size = len(tensor_dataset) - (train_size + val_size)

    print(f"Dataset sizes — total: {len(tensor_dataset)}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Split sizes: {[train_size, val_size, test_size]}")


    train_dataset, val_dataset, test_dataset = random_split(
        tensor_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4)

    def mean_pooling(last_hidden, mask):
        mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def contrastive_loss(q, p, temperature=0.05):
        q = F.normalize(q, dim=1)
        p = F.normalize(p, dim=1)
        logits = torch.matmul(q, p.T) / temperature # temperature-scaled cosine sim
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels) # actually softmax THEN cross-entropy internally 

    def should_stop(val_losses, patience=3, min_delta=0.0):
        best = min(val_losses)
        count = 0
        for loss in reversed(val_losses[:-1]):
            if loss - best > min_delta:
                count += 1
                if count >= patience:
                    return True
            else:
                break
        return False

    per_epoch_val_losses = []
    for epoch in range(8):
        val_losses = []
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for q_ids, q_mask, p_ids, p_mask in pbar:
            q_ids, q_mask = q_ids.to(device), q_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)

            with torch.cuda.amp.autocast():
                q_out = model(input_ids=q_ids, attention_mask=q_mask)
                p_out = model(input_ids=p_ids, attention_mask=p_mask)

                q_emb = mean_pooling(q_out.last_hidden_state, q_mask)
                p_emb = mean_pooling(p_out.last_hidden_state, p_mask)

                loss = contrastive_loss(q_emb, p_emb)

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            val_loss = 0
            val_count = 0
            if global_step % 200 == 0:
                model.eval()
                pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch}, validation")
                for q_ids, q_mask, p_ids, p_mask in pbar_val:
                    q_ids, q_mask = q_ids.to(device), q_mask.to(device)
                    p_ids, p_mask = p_ids.to(device), p_mask.to(device)            
                    with torch.inference_mode():
                        q_out = model(input_ids=q_ids, attention_mask=q_mask)
                        p_out = model(input_ids=p_ids, attention_mask=p_mask)

                        q_emb = mean_pooling(q_out.last_hidden_state, q_mask)
                        p_emb = mean_pooling(p_out.last_hidden_state, p_mask)

                        val_loss += contrastive_loss(q_emb, p_emb).item()
                        val_count += 1

                val_loss /= val_count
                val_losses.append(val_loss)
                print(f"[Step {global_step}] Validation Loss: {val_loss:.4f}")
                writer.add_scalar("val/loss", val_loss, global_step)
                model.train()

        avg_val_loss = np.mean(val_losses)
        per_epoch_val_losses.append(avg_val_loss)
        if should_stop(per_epoch_val_losses, patience=3):
            print("Early stopping triggered")
            break
        pbar.set_postfix(loss=loss.item())

        save_path = f"checkpoints/epoch_{epoch}.pt"
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # optionally also save loss, scheduler, etc.
            'loss': loss
        }, save_path)
        #torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
