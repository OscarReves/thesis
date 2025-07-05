from src.generator import get_generator

def main():
    # # import generator
    # generator = get_generator('suzume-llama3')
    # # extract logits
    # context = "The capital of France is in England."
    # prompt = "The capital of France is"
    # correct_answer="Paris"
    # incorrect_answer="London"
    # generator.check_if_flipped(context,prompt,correct_answer,incorrect_answer)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn.functional as F

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    alpha = 3.0

    context = "The capital of France is in England."
    prompt = "The capital of France is"

    context_ids = tokenizer.encode(context + prompt, return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits_with_context = model(context_ids).logits[:, -1, :]
        logits_without_context = model(prompt_ids).logits[:, -1, :]

    adjusted_logits = logits_with_context + alpha * (logits_with_context - logits_without_context)
    probs_orig = F.softmax(logits_without_context)
    probs_ctxt = F.softmax(logits_with_context)
    probs_cfg = F.softmax(adjusted_logits, dim=-1)

    for probs, lbl in [(probs_orig, "orig"), (probs_ctxt, "ctxt"), (probs_cfg, "cfg")]:
        print(f"---- {lbl} ----")
        top_k = 3
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
        print(f"Top {top_k} tokens:")
        for tok, p in zip(top_tokens, top_probs[0]):
            print(f"{tok!r} -> {p.item():.4f}")

        london_p = probs[0][tokenizer.encode("London")[1]]
        paris_p = probs[0][tokenizer.encode("Paris")[1]]
        print(f" > London: {london_p}")
        print(f" > Paris: {paris_p}")
        if london_p > paris_p:
            print(" > > FLIPPED!")

if __name__ == "__main__":
    main()