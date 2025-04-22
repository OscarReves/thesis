from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np

class GPT2Evaluator():
    def __init__(self, device, model_name="sshleifer/tiny-gpt2"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side= "left"
        self.tokenizer.chat_template = """
        {% for message in messages %}
        {{ message['role'] }}: {{ message['content'] }}
        {% endfor %}
        """ # added to make gpt2 compatible with apply_chat_template
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        print(f"Loaded model {model_name} on device {self.model.device}")

    def evaluate_answer(self, question, generated_answer, reference_answer, max_new_tokens=32):
        # this is a little hacky, but leave it for now 
        return self.evaluate_batch([question],[generated_answer],[reference_answer])
    
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
    
class NousHermesMistralEvaluator():
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
        print(f"Loaded model {model_name} on device {self.model.device}")

    def evaluate_answer(self, question, generated_answer, reference_answer, max_new_tokens=32):
        # this is a little hacky, but leave it for now 
        return self.evaluate_batch([question],[generated_answer],[reference_answer])
    
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
    
