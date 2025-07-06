from src.generator import get_generator

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn.functional as F
    from huggingface_hub import login
    from dotenv import load_dotenv
    import os

    # login required for gemma download 
    load_dotenv()  
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=hf_token)

    generator = get_generator('gemma-9b')

    model = generator.model
    tokenizer = generator.tokenizer
    device = model.device 

    system_prompt = (
        "You are a helpful assistant. You respond to questions in Danish.\n"
        "Respond briefly and accurately. Do not generate any extra questions or superfluous text.\n"
        "Be as concise as possible.\n"
    )

    def format_prompt_with_context(question, context, options):
        user_prompt = (
            "Givet konteksten, svar kun med bogstavet for den rigtige mulighed.\n"
            "#KONTEKST\n"
            f"{context}\n"
            "#SPØRGSMÅL\n"
            f"{question}\n"
            "#SVARMULIGHEDER\n"
            f"A: {options[0]}\n"
            f"B: {options[1]}\n"
            f"C: {options[2]}\n"
            #"#SVAR\n"
            #"Svaret er mulighed "
            )
        
        
        messages = [
            {"role": "user", "content": user_prompt},
        ]   

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def format_prompt_no_context(question, options):
        user_prompt = (
            "Svar kun med bogstavet for den rigtige mulighed.\n"
            "#SPØRGSMÅL\n"
            f"{question}\n"
            "#SVARMULIGHEDER\n"
            f"A: {options[0]}\n"
            f"B: {options[1]}\n"
            f"C: {options[2]}\n"
            #"#SVAR\n"
            #"Svaret er mulighed "
            )
        
        
        messages = [
            {"role": "user", "content": user_prompt},
        ]   

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    from src.utils import load_questions_by_type
    dataset = load_questions_by_type(path=None, type='citizenship')

    question = dataset[0]['question']
    options = dataset[0]['options']
    context = "Medlemmer må kun vælges for en periode ad gangen."


    def check_if_flipped_mc(model, tokenizer, context, question, options, correct_answer, incorrect_answer, alpha=3.0):

        context_prompt = format_prompt_with_context(question=question, context=context, options=options)
        print(f"Context prompt:\n{context_prompt}")

        no_context_prompt = format_prompt_no_context(question=question, options=options)
        print(f"No context prompt:\n{no_context_prompt}")

        context_ids = tokenizer.encode(context_prompt, return_tensors="pt").to(device)
        prompt_ids = tokenizer.encode(no_context_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits_with_context = model(context_ids).logits[:, -1, :]
            logits_without_context = model(prompt_ids).logits[:, -1, :]

        adjusted_logits = logits_with_context + alpha * (logits_with_context - logits_without_context)
        probs_orig = F.softmax(logits_without_context, dim=-1)
        probs_ctxt = F.softmax(logits_with_context, dim=-1)
        probs_cfg = F.softmax(adjusted_logits, dim=-1)

        for probs, lbl in [(probs_orig, "orig"), (probs_ctxt, "ctxt"), (probs_cfg, "cfg")]:
            print(f"---- {lbl} ----")
            top_k = 3
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
            print(f"Top {top_k} tokens:")
            for tok, p in zip(top_tokens, top_probs[0]):
                print(f"{tok!r} -> {p.item():.4f}")

            correct_p = probs[0][tokenizer.encode(correct_answer)[1]]
            incorrect_p = probs[0][tokenizer.encode(incorrect_answer)[1]]
            print(f" > {correct_answer}: {correct_p}")
            print(f" > {incorrect_answer}: {incorrect_p}")
            if incorrect_p > correct_p:
                print(" > > FLIPPED!")

    print(f"device = {device}")
    #model.to(device)
    check_if_flipped_mc(model, tokenizer, context, question, options, correct_answer="A", incorrect_answer="B")



if __name__ == "__main__":
    main()