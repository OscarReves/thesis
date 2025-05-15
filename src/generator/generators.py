from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import os 

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

# class NousHermesMistral2Generator():
#     # fix the naming scheme and start working with inheritance 
#     def __init__(self, model_name="NousResearch/Nous-Hermes-2-Mistral-7B-DPO"):
#         model_path = 'models/nous-hermes'
#         print("Loading model from local directory...")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             local_files_only=True
#         )

#     def generate_answer(self, question, context, max_new_tokens=128):
#         # this is a little hacky, but leave it for now 
#         return self.generate_batch([question],[context], max_new_tokens=max_new_tokens)
    
#     def generate_batch(self, questions, contexts, max_new_tokens=128):
#         system_prompt = (
#             "You are a helpful assistant. You respond to questions in Danish. "
#             "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
#             "Be as concise as possible."
#         )

#         prompts = [
#             self.tokenizer.apply_chat_template([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Besvar følgende spørgsmål ud fra kontekst:\nKontekst: {c}\nSpørgsmål: {q}"}
#             ], tokenize=False, add_generation_prompt=True)
#             for q, c in zip(questions, contexts)
#         ]

#         inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 num_beams=1,
#                 use_cache=True,
#                 #temperature=0.7,
#                 #top_p=0.9,
#                 eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         input_lengths = [len(i) for i in inputs["input_ids"]]
#         outputs = [
#             self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
#             for i, input_len in enumerate(input_lengths)
#         ]
#         return outputs
    
#     def generate_batch_mc(self, questions, contexts, options, max_new_tokens=128):
#         # Answer citizenship test multiple choice questions in batches
#         system_prompt = (
#             "You are a helpful assistant. You respond to questions in Danish. "
#             "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
#             "Be as concise as possible. "
#             "The context may or may not be relevant."
#         )

#         user_prompts = [
#             (
#                 "Givet konteksten, svar kun med bogstavet for den rigtige mulighed."
#                 "#KONTEKST"
#                 f"{c}"
#                 "#SPØRGSMÅL"
#                 f"{q}"
#                 "#SVARMULIGHEDER"
#                 f"A: {o[0]}"
#                 f"B: {o[1]}"
#                 f"C: {o[2]}"
#                 "#SVAR"
#                 "Svaret er mulighed "
#             )
#             for q, c, o in zip(questions, contexts, options)
#         ]

#         prompts = [
#             self.tokenizer.apply_chat_template([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": up}
#             ], tokenize=False, add_generation_prompt=True)
#             for up in user_prompts
#         ]

#         inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 num_beams=1,
#                 use_cache=True,
#                 #temperature=0.7,
#                 #top_p=0.9,
#                 eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         input_lengths = [len(i) for i in inputs["input_ids"]]
#         outputs = [
#             self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
#             for i, input_len in enumerate(input_lengths)
#         ]
#         return outputs

#     def generate_batch_mc_no_context(self, questions, options, max_new_tokens=128):
#             # Answer citizenship test multiple choice questions in batches
#             system_prompt = (
#                 "You are a helpful assistant. You respond to questions in Danish. "
#                 "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
#                 "Be as concise as possible. "
#                 "The context may or may not be relevant."
#             )

#             user_prompts = [
#                 (
#                     "Svar kun med bogstavet for den rigtige mulighed."
#                     "#SPØRGSMÅL"
#                     f"{q}"
#                     "#SVARMULIGHEDER"
#                     f"A: {o[0]}"
#                     f"B: {o[1]}"
#                     f"C: {o[2]}"
#                     "#SVAR"
#                     "Svaret er mulighed "
#                 )
#                 for q, o in zip(questions, options)
#             ]

#             prompts = [
#                 self.tokenizer.apply_chat_template([
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": up}
#                 ], tokenize=False, add_generation_prompt=True)
#                 for up in user_prompts
#             ]

#             inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

#             with torch.inference_mode():
#                 output_ids = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     do_sample=False,
#                     num_beams=1,
#                     use_cache=True,
#                     #temperature=0.7,
#                     #top_p=0.9,
#                     eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )

#             input_lengths = [len(i) for i in inputs["input_ids"]]
#             outputs = [
#                 self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
#                 for i, input_len in enumerate(input_lengths)
#             ]
#             return outputs


#     def generate_batch_no_context(self, questions, max_new_tokens=128):
#         system_prompt = (
#             "You are a helpful assistant. You respond to questions in Danish. "
#             "Respond briefly and accurately. Do not generate any extra questions or superfluous text. "
#             "Be as concise as possible."
#         )

#         prompts = [
#             self.tokenizer.apply_chat_template([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"{q}"}
#             ], tokenize=False, add_generation_prompt=True)
#             for q in questions
#         ]

#         inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 num_beams=1,
#                 use_cache=True,
#                 #temperature=0.7,
#                 #top_p=0.9,
#                 eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         input_lengths = [len(i) for i in inputs["input_ids"]]
#         outputs = [
#             self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
#             for i, input_len in enumerate(input_lengths)
#         ]
#         return outputs

#     def rewrite_questions(self, questions, contexts, max_new_tokens=256):
#         system_prompt = (
#             "You are a helpful assistant. You respond to prompts in Danish. "
#             "Respond briefly and accurately."
#             "Be as concise as possible."
#         )

#         prompts = [
#             self.tokenizer.apply_chat_template([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"""
#                 You are an expert question disambiguator.
#                 Using the context, rewrite the following question so that it would be unambiguous in the absence of the context.
#                 The question should be answerable as a stand-alone question, but should not include the answer. 

#                 Context: {c}
                 
#                 Question: {q}
#                 """}
#             ], tokenize=False, add_generation_prompt=True)
#             for q, c in zip(questions, contexts)
#         ]

#         inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 num_beams=1,
#                 use_cache=True,
#                 #temperature=0.7,
#                 #top_p=0.9,
#                 eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         input_lengths = [len(i) for i in inputs["input_ids"]]
#         outputs = [
#             self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True).strip()
#             for i, input_len in enumerate(input_lengths)
#         ]
#         return outputs

class BaseGenerator:
    def __init__(self, model_name, save_name = None):
        base_path = "/dtu/p1/oscrev/models"
        model_path = os.path.join(base_path, save_name)

        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name} from Hugging Face...")
            os.makedirs(model_path, exist_ok=True)
            # Download and save
            AutoTokenizer.from_pretrained(model_name).save_pretrained(model_path)
            AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(model_path)

        print(f"Loading model {model_name} from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

    def generate_answer(self, question, context, max_new_tokens=128):
        # this is a little hacky, but leave it for now 
        return self.generate_batch([question],[context], max_new_tokens=max_new_tokens)
    
    def generate_batch(self, questions, contexts, max_new_tokens=128):
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

    def rewrite_questions(self, questions, contexts, max_new_tokens=256):
        system_prompt = (
            "You are a helpful assistant. You respond to prompts in Danish. "
            "Respond briefly and accurately."
            "Be as concise as possible."
        )

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
                You are an expert question disambiguator.
                Using the context, rewrite the following question so that it would be unambiguous in the absence of the context.
                The question should be answerable as a stand-alone question, but should not include the answer. 

                Context: {c}
                 
                Question: {q}
                """}
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
    
class NousHermesMistralGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(
            model_name="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
            save_name="nous-hermes-mistral"
            )

class SuzumeLlama3Generator(BaseGenerator):
    def __init__(self):
        super().__init__("lightblue/suzume-llama-3-8B-multilingual")

class Yi34BGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("01-ai/Yi-34B-Chat")