from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from tqdm import tqdm
import numpy as np
import os 
import gc
import sys
    
class BaseGenerator:
    def __init__(self, model_name, save_name = None):
        base_path = "/dtu/p1/oscrev/models"
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
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.save_pretrained(model_path)

        else:
            print(f"Loading model {model_name} from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,trust_remote_code=True)
            self.tokenizer.padding_side = "left"
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

        return self.generate_from_prompts(prompts=prompts)
    

    def get_mc_prompt_no_context(self, questions, options):
        user_prompts = [
            (
                "Svar kun med bogstavet for den rigtige mulighed.\n"
                "#SPØRGSMÅL\n"
                f"{q}\n"
                "#SVARMULIGHEDER\n"
                f"A: {o[0]}\n"
                f"B: {o[1]}\n"
                f"C: {o[2]}\n"
                "#SVAR\n"
                "Svaret er mulighed "
            )
            for q, o in zip(questions, options)
        ]

        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": up}
            ], tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]
        return prompts
    
    def get_mc_prompt(self, questions, contexts, options):
        user_prompts = [
            (
                "Givet konteksten, svar kun med bogstavet for den rigtige mulighed.\n"
                "#KONTEKST\n"
                f"{c}\n"
                "#SPØRGSMÅL\n"
                f"{q}\n"
                "#SVARMULIGHEDER\n"
                f"A: {o[0]}\n"
                f"B: {o[1]}\n"
                f"C: {o[2]}\n"
                "#SVAR\n"
                "Svaret er mulighed "
            )
            for q, c, o in zip(questions, contexts, options)
        ]
        prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": up}
            ], tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]
        return prompts
    


    def cfg_answer(self, questions, contexts, options, alpha=0.1):
        
        mc_no_context_prompt = self.get_mc_prompt_no_context(questions, options)
        mc_prompt = self.get_mc_prompt(questions, contexts, options)
        
        logits_no_context = self.get_logits(mc_no_context_prompt)[:, -1, :]
        logits = self.get_logits(mc_prompt)[:, -1, :]

        print("logits full shape:", logits.shape)  # should be [batch, seq_len, vocab]
        print("logits_no_context full shape:", logits_no_context.shape)

        # for a, b in zip(mc_prompt, mc_no_context_prompt):
        #     print("With context:", repr(a[-80:]))
        #     print("No context: ", repr(b[-80:]))
        #     print("---")


        # self.find_nan_prompt(mc_no_context_prompt)        
        # self.find_nan_prompt(mc_prompt)

        # self.trace_nan_tokens(mc_no_context_prompt)

        # sys.stdout.flush()

        # assert logits.shape == logits_no_context.shape, "Shape mismatch in logits vs. logits_no_context"
        # assert not torch.isnan(logits).any(), "NaNs in logits"
        # assert not torch.isnan(logits_no_context).any(), "NaNs in logits_no_context"

        cfg_logits = logits + alpha*(logits-logits_no_context)
        
        print("cfg_logits min:", cfg_logits.min().item())
        print("cfg_logits max:", cfg_logits.max().item())


        res =  {
            'answers' : self.decode_logits(logits),
            'guided_answers' : self.decode_logits(cfg_logits)
            }
        
        return res


    def decode_logits(self, logits):
        token_ids = torch.argmax(logits, dim=-1)
        tokens = self.tokenizer.batch_decode(token_ids.unsqueeze(1))
        return tokens
    
    def get_logits_old(self, prompts):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self.model.config.max_position_embeddings
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        #return outputs.logits[:, -1, :] # extract last logit (equivalent to next token)
        return outputs.logits

    def get_logits(self, prompts):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self.model.config.max_position_embeddings
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,                 # turn on score recording
                return_dict_in_generate=True
            )
        #return outputs.logits[:, -1, :] # extract last logit (equivalent to next token)
        logits  = torch.stack(out.scores) 
        return logits

    def find_nan_prompt(self, prompts):
        for i, prompt in enumerate(prompts):
            max_length = self.model.config.max_position_embeddings
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            if torch.isnan(outputs.logits).any():
                print("⚠️ NaNs in example:", i)
                print(prompt)
                break

    def trace_nan_tokens(self, prompt):
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=getattr(self.model.config, "max_position_embeddings", 2048)).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits  # (1, seq_len, vocab_size)
        nan_mask = torch.isnan(logits)  # (1, seq_len, vocab_size)
        nan_positions = nan_mask.any(dim=-1)[0]  # (seq_len,)

        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        print("\n🧪 NaN token trace:")
        for i, (tok, is_nan) in enumerate(zip(tokens, nan_positions)):
            flag = "⚠️" if is_nan.item() else "  "
            print(f"{flag} Token {i:>3}: {tok}")

        if nan_positions.any():
            print("\n⚠️ NaNs detected in logits.")
        else:
            print("\n✅ No NaNs detected.")



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