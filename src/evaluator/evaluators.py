from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np

def GPT2Evaluator():
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def evaluate_answer(self, question, context, max_new_tokens=32):
        # this is a little hacky, but leave it for now 
        return self.generate_batch([question],[context])
    
    def evaluate_batch(self, questions, generated_answers, reference_answers, max_new_tokens=256):
        system_prompt = (
            "You are a helpful assistant."
            "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
            "Be as concise as possible."
        )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
            You are an expert evaluator. Compare the following generated answer to the reference answer, and rate how well it answers the question on a scale of 1 to 5.

            Question: {q}

            Generated Answer: {ga}

            Reference Answer: {ra}

            Rate the quality (1–5) and briefly explain your reasoning.
            """
            }
            ], tokenize=False, add_generation_prompt=True)
            for q, ga, ra in zip(questions, generated_answers, reference_answers)
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