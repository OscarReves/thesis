from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F
import torch._dynamo
from tqdm import tqdm
import numpy as np
import os 

    
class BaseGenerator:
    def __init__(self, model_name, save_name = None, remote=True):
        if remote:
            base_path = "/dtu/p1/oscrev/models"
        else:
            base_path = 'models'
        
        
        model_path = os.path.join(base_path, save_name)
        
        # self.eos_token = "<|im_end|>" # also shouldn't be necesary? All cauusal models should have eos pre-defined
        self.system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish.\n"
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text.\n"
            "Be as concise as possible.\n"
        )

        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name} from Hugging Face...")
            os.makedirs(model_path, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
            self.tokenizer.save_pretrained(model_path)
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.save_pretrained(model_path)

        else:
            print(f"Loading model {model_name} from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,trust_remote_code=True)
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            self.model.eval()
            # enable TF32
            torch.set_float32_matmul_precision('high')



    def generate_answer(self, question, context, max_new_tokens=128):
        # this is a little hacky, but leave it for now 
        return self.generate_batch([question],[context], max_new_tokens=max_new_tokens)
    
    def generate_batch(self, questions, contexts, max_new_tokens=128):
        # system_prompt = (
        #     "You are a helpful assistant. You respond to questions in Danish. "
        #     "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
        #     "Be as concise as possible."
        # )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Besvar følgende spørgsmål ud fra kontekst:\nKontekst: {c}\nSpørgsmål: {q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q, c in zip(questions, contexts)
        ]

        return self.generate_from_prompts(prompts=prompts)
    
    def generate_batch_mc(self, questions, contexts, options, max_new_tokens=128):
        # Answer citizenship test multiple choice questions in batches
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible. "
            "The context may or may not be relevant."
        )

        user_prompts = [
            (
                "Givet konteksten, svar kun med bogstavet for den rigtige mulighed."
                "#KONTEKST"
                f"{c}"
                "#SPØRGSMÅL"
                f"{q}"
                "#SVARMULIGHEDER"
                f"A: {o[0]}"
                f"B: {o[1]}"
                f"C: {o[2]}"
                "#SVAR"
                "Svaret er mulighed "
            )
            for q, c, o in zip(questions, contexts, options)
        ]

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up}
            ], tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]

        return self.generate_from_prompts(prompts=prompts)

    def generate_batch_mc_no_context(self, questions, options, max_new_tokens=128):
            # Answer citizenship test multiple choice questions in batches
            system_prompt = (
                "You are a helpful assistant. You respond to questions in Danish. "
                "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
                "Be as concise as possible. "
                "The context may or may not be relevant."
            )

            user_prompts = [
                (
                    "Svar kun med bogstavet for den rigtige mulighed."
                    "#SPØRGSMÅL"
                    f"{q}"
                    "#SVARMULIGHEDER"
                    f"A: {o[0]}"
                    f"B: {o[1]}"
                    f"C: {o[2]}"
                    "#SVAR"
                    "Svaret er mulighed "
                )
                for q, o in zip(questions, options)
            ]

            prompts = [
                self.tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": up}
                ], tokenize=False, add_generation_prompt=True)
                for up in user_prompts
            ]

            return self.generate_from_prompts(prompts=prompts)

    def generate_batch_no_context(self, questions, max_new_tokens=128):
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible."
        )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q in questions
        ]

        return self.generate_from_prompts(prompts=prompts)


    def generate_from_prompts(self, prompts, max_new_tokens=256):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                #temperature=0.7,
                #top_p=0.9,
                #eos_token_id=self.tokenizer.convert_tokens_to_ids(self.eos_token), # this shouldn't be necessary?
                #pad_token_id=self.tokenizer.eos_token_id # same for this?
            )

        input_lengths = [len(i) for i in inputs["input_ids"]]
        outputs = [
            self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
            for i, input_len in enumerate(input_lengths)
        ]
        return outputs

    def check_if_flipped(self, context, prompt, correct_answer, incorrect_answer, alpha=3.0):

        model = self.model
        device = model.device
        tokenizer = self.tokenizer

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

            correct_p = probs[0][tokenizer.encode(correct_answer)[1]]
            incorrect_p = probs[0][tokenizer.encode(incorrect_answer)[1]]
            print(f" > {correct_answer}: {correct_p}")
            print(f" > {incorrect_answer}: {incorrect_p}")
            if incorrect_p > correct_p:
                print(" > > FLIPPED!")


    def format_prompt_with_context(self, context, prompt, truncated_answer_sentence):
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot.",
            },
            {"role": "user", "content": context + " " + prompt},
            {"role": "assistant", "content": truncated_answer_sentence}
        ]   

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        return prompt


    def format_prompt_no_context(self, prompt, truncated_answer_sentence):
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": truncated_answer_sentence}
        ]   

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        return prompt
    
    def check_if_flipped_with_chat_template(self, context, prompt, correct_answer, incorrect_answer, alpha=3.0):

        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        context_prompt = self.format_prompt_with_context(context, prompt, truncated_answer_sentence=prompt)
        no_context_prompt = self.format_prompt_no_context(prompt, truncated_answer_sentence=prompt)

        context_ids = tokenizer.encode(context_prompt, return_tensors="pt").to(device)
        prompt_ids = tokenizer.encode(no_context_prompt, return_tensors="pt").to(device)

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

            correct_p = probs[0][tokenizer.encode(correct_answer)[1]]
            incorrect_p = probs[0][tokenizer.encode(incorrect_answer)[1]]
            print(f" > {correct_answer}: {correct_p}")
            print(f" > {incorrect_answer}: {incorrect_p}")
            if incorrect_p > correct_p:
                print(" > > FLIPPED!")

    def format_prompt_with_context_mc(self, question, context, options):
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
            # {
            #     "role": "system",
            #     "content": self.system_prompt,
            # },
            {"role": "user", "content": user_prompt},
        ]   

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def format_prompt_no_context_mc(self, question, options):
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
            # {
            #     "role": "system",
            #     "content": self.system_prompt,
            # },
            {"role": "user", "content": user_prompt},
        ]   

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    

    def check_if_flipped_mc(self, context, question, options, correct_answer, incorrect_answer, alpha=3.0):

        model = self.model
        device = self.model.device
        tokenizer = self.tokenizer

        context_prompt = self.format_prompt_with_context_mc(question=question, context=context, options=options)
        print(f"Context prompt:\n{context_prompt}")

        no_context_prompt = self.format_prompt_no_context_mc(question=question, options=options)
        print(f"No context prompt:\n{no_context_prompt}")

        context_ids = tokenizer.encode(context_prompt, return_tensors="pt").to(device)
        prompt_ids = tokenizer.encode(no_context_prompt, return_tensors="pt").to(device)

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

            # tok_correct = tokenizer(correct_answer, add_special_tokens=False).input_ids[0]
            # tok_incorrect  = tokenizer(incorrect_answer,  add_special_tokens=False).input_ids[0]
            # correct_p = probs[0, tok_correct]
            # incorrect_p  = probs[0, tok_incorrect]
            correct_p = probs[0][tokenizer.encode(correct_answer)[1]]
            incorrect_p = probs[0][tokenizer.encode(incorrect_answer)[1]]
            print(f" > {correct_answer}: {correct_p}")
            print(f" > {incorrect_answer}: {incorrect_p}")
            if incorrect_p > correct_p:
                print(" > > FLIPPED!")

    def generate_answer_cfg(self, context, question, options, alpha = 3.0, silent=True):

        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        context_prompt = self.format_prompt_with_context_mc(question=question, context=context, options=options)
        if not silent:
            print(f"Context prompt:\n{context_prompt}")

        no_context_prompt = self.format_prompt_no_context_mc(question=question, options=options)
        if not silent:
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

        if not silent:
            for probs, lbl in [(probs_orig, "orig"), (probs_ctxt, "ctxt"), (probs_cfg, "cfg")]:
                print(f"---- {lbl} ----")
                top_k = 3
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
                print(f"Top {top_k} tokens:")
                for tok, p in zip(top_tokens, top_probs[0]):
                    print(f"{tok!r} -> {p.item():.4f}")

        if silent:
            probs = probs_cfg
            top_k = 3
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]

        return top_tokens[0]

    # def get_logits(self, prompt):
    #     token_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
    #     with torch.no_grad():
    #         logits = self.model(token_ids).logits[:, -1, :]
        
    #     return logits
    
    # def decode_logits(self, logits, top_k=1):
    #     # return top-k tokens 
    #     probs = F.softmax(logits, dim=-1)
    #     top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
    #     top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices[0]]
    #     return top_tokens

    # def get_logits(self, prompts):
    #     """
    #     Args
    #     ----
    #     prompts : str | list[str]  
    #         Single prompt or a batch of prompts.

    #     Returns
    #     -------
    #     torch.Tensor  # (batch, vocab)
    #         Logits for the *last* non-padding token in each prompt.
    #     """
    #     if isinstance(prompts, str):
    #         prompts = [prompts]

    #     # Batch-encode with left-padding (or whatever your model expects)
    #     enc = self.tokenizer(
    #         prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=False,
    #     ).to(self.model.device)

    #     with torch.no_grad():
    #         all_logits = self.model(**enc).logits         # (B, L, V)

    #     # Index logits at each sequence’s final (non-pad) position
    #     seq_ends = (enc["input_ids"] != self.tokenizer.pad_token_id).sum(1) - 1
    #     batch_idx = torch.arange(len(prompts), device=seq_ends.device)
    #     last_logits = all_logits[batch_idx, seq_ends]     # (B, V)

    #     return last_logits

    # def decode_logits(self, logits, top_k=1):
    #     """
    #     Args
    #     ----
    #     logits : torch.Tensor  # (batch, vocab)
    #     top_k  : int

    #     Returns
    #     -------
    #     list[list[str]]
    #         Top-k decoded tokens for each element in the batch.
    #     """
    #     probs = F.softmax(logits, dim=-1)                 # (B, V)
    #     _, top_idx = torch.topk(probs, top_k, dim=-1)      # (B, k)

    #     return [
    #         [self.tokenizer.decode([idx.item()]) for idx in row]
    #         for row in top_idx
    #     ]
    def get_logits(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]

        # Make sure tokenizer is set for left padding
        self.tokenizer.padding_side = "left"

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)                 # (B, L)

        with torch.inference_mode():
            logits = self.model(**enc, use_cache=False).logits   # (B, L, V)

        return logits[:, -1, :]                 # (B, V)

    def decode_logits(self, logits, top_k=1):
        probs = F.softmax(logits, dim=-1)               # (B, V)
        _, top_idx = torch.topk(probs, top_k, dim=-1)   # (B, k)

        return [
            [self.tokenizer.decode([idx.item()]) for idx in row]
            for row in top_idx
        ]
    
    # def cfg_batch(self, contexts, questions, options, alpha, silent = True):
    #     model = self.model
    #     device = model.device
    #     tokenizer = self.tokenizer

    #     context_prompts = []
    #     no_context_prompts = []

    #     for question, context, option in zip(questions, contexts, options):
    #         context_prompt = self.format_prompt_with_context_mc(question=question, context=context, options=option)
    #         if not silent:
    #             print(f"Context prompt:\n{context_prompt}")
    #         context_prompts.append(context_prompt)

    #         no_context_prompt = self.format_prompt_no_context_mc(question=question, options=option)
    #         if not silent:
    #             print(f"No context prompt:\n{no_context_prompt}")
    #         no_context_prompts.append(no_context_prompt)
        
    #     logits_with_context = self.get_logits(context_prompts)
    #     logits_without_context = self.get_logits(no_context_prompts)
    #     adjusted_logits = logits_with_context + alpha * (logits_with_context - logits_without_context)
        

    #     answers = {
    #         'no_context_answers' : self.decode_logits(logits_without_context),
    #         'cfg_answers' : self.decode_logits(adjusted_logits)
    #         }
        
    #     return answers
    @torch.inference_mode()
    def cfg_batch(self, contexts, questions, options, alpha, reference_answers, retrieval_scores=None, silent=True):
        # 1. build both prompt variants first
        ctx_prompts, noc_prompts = [], []
        for q, c, opt in zip(questions, contexts, options):
            ctx_prompts.append(self.format_prompt_with_context_mc(q, c, opt))
            noc_prompts.append(self.format_prompt_no_context_mc(q, opt))

        # 2. single forward pass (batch = 2 * len(questions))
        all_prompts = ctx_prompts + noc_prompts
        all_logits = self.get_logits(all_prompts)            # (2B, V)

        B = len(questions)
        logits_ctx, logits_noc = all_logits[:B], all_logits[B:]

        # 3. classifier-free guidance
        if retrieval_scores is not None:
            alpha = torch.tensor(3.31529067 * retrieval_scores -1.3410987260332883 - 0.1).to(self.model.device) # dynamic alpha 
            print(f"Aphas = {alpha}")
        adjusted = logits_ctx + alpha * (logits_ctx - logits_noc)

        cfg_answers = self.decode_logits(adjusted)
        no_context_answers = self.decode_logits(logits_noc)
        answers_with_context = self.decode_logits(logits_ctx)

        # 4. solve for guidance scale algebraically 
        alphas = self.solve_for_alpha(
            answers_with_context=answers_with_context,
            reference_answers=reference_answers,
            logits_with_context=logits_ctx,
            cfg_logits=adjusted,
            no_context_logits=logits_noc,
            no_context_answers=no_context_answers
        )

        

        return {
            "no_context_answers":   no_context_answers,
            "answers_with_context": answers_with_context,
            "cfg_answers":          cfg_answers,
            "alphas":               alphas,
            #"logits_ctx":           logits_ctx,
            #"logits_noc":           logits_noc 
        }
    
    def solve_for_alpha(self, answers_with_context, reference_answers, logits_with_context, cfg_logits, no_context_logits, no_context_answers):
        # probably there is a way to vectorize this. Maybe not necessary?
        alphas = []
        for ans, ref, logits_ctx, logits_cfg, logits_n, ans_n in zip(
            answers_with_context, reference_answers, logits_with_context, cfg_logits, no_context_logits, no_context_answers): 
            if ans[0] == ref[0]:
                #alpha = 0.0
                #alphas.append(alpha)
                # get token ids 
                if ans_n[0] != ref[0]: # if answer without context is wrong
                    ref_idx = self.tokenizer.encode(ref[0])[1]
                    ans_n_idx = self.tokenizer.encode(ans_n[0])[1]
                    
                    # get logits 
                    tc = logits_with_context[0][ref_idx] # indexed incorrectly? 
                    fc = logits_with_context[0][ans_n_idx]
                    tcfg = cfg_logits[0][ref_idx]
                    fcfg = cfg_logits[0][ans_n_idx]

                    # solve for alpha required to flip answer
                    neg_beta = (fc - tc) / (tcfg-fcfg) # this method ignores the third answer (and all other logits)
                    alpha = neg_beta - 1
                    alphas.append(alpha.item()) # cast to float

                else:
                    alpha = -1.0
                    alphas.append(alpha)


            else:
                # get token ids 
                ref_idx = self.tokenizer.encode(ref[0])[1]
                ans_idx = self.tokenizer.encode(ans[0])[1]
                
                # get logits 
                tc = logits_with_context[0][ref_idx] # indexed incorrectly? 
                fc = logits_with_context[0][ans_idx]
                tcfg = cfg_logits[0][ref_idx]
                fcfg = cfg_logits[0][ans_idx]

                # solve for alpha required to flip answer
                alpha = (fc - tc) / (tcfg-fcfg) # this method ignores the third answer (and all other logits)
                alphas.append(alpha.item()) # cast to float

        return alphas

class TinyLlamaGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            save_name="tinyllama",
            remote=False
            )

class NousHermesMistralGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
            save_name="nous-hermes-mistral"
            )

class SuzumeLlama3Generator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="lightblue/suzume-llama-3-8B-multilingual",
            save_name="suzume-llama3"
            )

class Yi34BGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="01-ai/Yi-34B-Chat",
            save_name="yi-34b"
            )
        
class SnakModelGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="NLPnorth/snakmodel-7b-instruct",
            save_name="snakmodel"
            )
        #self.eos_token = self.tokenizer.eos_token
        self.system_prompt = "Du er Snakmodel, skabt af IT-Universitetet i København. Du er en hjælpsom assistent."


class Gemma9bGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="google/gemma-2-9b-it",
            save_name="gemma-2-9b-it"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id


    # requires its own prompt formatting since the 'system' role is not supported in chat_template    
    def generate_batch(self, questions, contexts, max_new_tokens=32):
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. " 
            "Be as concise as possible."
            "The context may or may not be relevant."
        )

        # system_prompt = (
        #     "You are a helpful assistant. You respond to questions in Danish. "
        #     "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
        #     "Be as concise as possible."
        # )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": f"{system_prompt}\nBesvar følgende spørgsmål ud fra kontekst:\nKontekst: {c}\nSpørgsmål: {q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q, c in zip(questions, contexts)
        ]

        return self.generate_from_prompts(prompts=prompts)

    def generate_batch_mc(self, questions, contexts, options, max_new_tokens=32):
        # Answer citizenship test multiple choice questions in batches
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible. "
            "The context may or may not be relevant."
        )

        user_prompts = [
            (
                "Givet konteksten, svar kun med bogstavet for den rigtige mulighed."
                "#KONTEKST"
                f"{c}"
                "#SPØRGSMÅL"
                f"{q}"
                "#SVARMULIGHEDER"
                f"A: {o[0]}"
                f"B: {o[1]}"
                f"C: {o[2]}"
                "#SVAR"
                "Svaret er mulighed "
            )
            for q, c, o in zip(questions, contexts, options)
        ]

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": system_prompt + "\n" + up}
            ], tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]

        return self.generate_from_prompts(prompts=prompts)

    def generate_batch_mc_no_context(self, questions, options, max_new_tokens=32):
            # Answer citizenship test multiple choice questions in batches
            system_prompt = (
                "You are a helpful assistant. You respond to questions in Danish. "
                "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
                "Be as concise as possible. "
                "The context may or may not be relevant."
            )

            user_prompts = [
                (
                    "Svar kun med bogstavet for den rigtige mulighed."
                    "#SPØRGSMÅL"
                    f"{q}"
                    "#SVARMULIGHEDER"
                    f"A: {o[0]}"
                    f"B: {o[1]}"
                    f"C: {o[2]}"
                    "#SVAR"
                    "Svaret er mulighed "
                )
                for q, o in zip(questions, options)
            ]

            prompts = [
                self.tokenizer.apply_chat_template([
                    {"role": "user", "content": system_prompt + "\n" + up}
                ], tokenize=False, add_generation_prompt=True)
                for up in user_prompts
            ]

            return self.generate_from_prompts(prompts=prompts)

    def generate_batch_no_context(self, questions, max_new_tokens=32):
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible."
        )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": f"{system_prompt} \n{q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q in questions
        ]

        return self.generate_from_prompts(prompts=prompts)
    
    def get_mc_prompt_no_context(self, questions, options):
        user_prompts = [
            (
                "Svar kun med bogstavet for den rigtige mulighed."
                "#SPØRGSMÅL"
                f"{q}"
                "#SVARMULIGHEDER"
                f"A: {o[0]}"
                f"B: {o[1]}"
                f"C: {o[2]}"
                "#SVAR"
                "Svaret er mulighed "
            )
            for q, o in zip(questions, options)
        ]

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": self.system_prompt + "\n" + up}
            ], tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]
        return prompts
    
    def get_mc_prompt(self, questions, contexts, options):
        user_prompts = [
            (
                "Givet konteksten, svar kun med bogstavet for den rigtige mulighed."
                "#KONTEKST"
                f"{c}"
                "#SPØRGSMÅL"
                f"{q}"
                "#SVARMULIGHEDER"
                f"A: {o[0]}"
                f"B: {o[1]}"
                f"C: {o[2]}"
                "#SVAR"
                "Svaret er mulighed "
            )
            for q, c, o in zip(questions, contexts, options)
        ]
        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": self.system_prompt + "\n" + up}
            ], tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]
        return prompts
    


class Gemma9bGeneratorNewPrompt(Gemma9bGenerator):
    def __init__(self):
        super().__init__()
    
    def generate_batch(self, questions, contexts, max_new_tokens=32):
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. " 
            "Be as concise as possible."
            "The context may or may not be relevant."
            "If the context is not relevant, simply respond from memory."
        )

        # system_prompt = (
        #     "You are a helpful assistant. You respond to questions in Danish. "
        #     "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
        #     "Be as concise as possible."
        # )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "user", "content": f"{system_prompt}\nBesvar følgende spørgsmål ud fra kontekst:\nKontekst: {c}\nSpørgsmål: {q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q, c in zip(questions, contexts)
        ]

        return self.generate_from_prompts(prompts=prompts)