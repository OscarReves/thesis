from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np

class GPT2Generator():
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_answer(self, question, context, max_new_tokens=32):
        # this is a little hacky, but leave it for now 
        return self.generate_batch([question],[context])
    
    def generate_batch(self, questions, contexts, max_new_tokens=32):
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible."
        )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Besvar følgende spørgsmål ud fra kontektst:\nKontekst: {c}\nSpørgsmål: {q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q, c in zip(questions, contexts)
        ]

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
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_lengths = [len(i) for i in inputs["input_ids"]]
        outputs = [
            self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
            for i, input_len in enumerate(input_lengths)
        ]
        return outputs

class NousHermesMistral2Generator():
    # fix the naming scheme and start working with inheritance 
    def __init__(self, model_name="NousResearch/Nous-Hermes-2-Mistral-7B-DPO"):
        model_path = 'models/nous-hermes'
        print("Loading model from local directory...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

    def generate_answer(self, question, context, max_new_tokens=64):
        # this is a little hacky, but leave it for now 
        return self.generate_batch([question],[context], max_new_tokens=max_new_tokens)
    
    def generate_batch(self, questions, contexts, max_new_tokens=64):
        system_prompt = (
            "You are a helpful assistant. You respond to questions in Danish. "
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible."
        )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Besvar følgende spørgsmål ud fra kontekst:\nKontekst: {c}\nSpørgsmål: {q}"}
            ], tokenize=False, add_generation_prompt=True)
            for q, c in zip(questions, contexts)
        ]

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
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_lengths = [len(i) for i in inputs["input_ids"]]
        outputs = [
            self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
            for i, input_len in enumerate(input_lengths)
        ]
        return outputs
