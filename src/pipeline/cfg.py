import json
from tqdm import tqdm
import time
from src.utils import save_to_json

def test_cfg(question_dataset, retriever, generator, save_path, alpha=3.0,
             batch_size = 1, silent=True, max_samples=None):
    
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    
    correct = 0

    for i in tqdm(range(0, len(question_dataset)), 
                  desc=f"Answering questions in batches of {1}"):
        question = question_dataset['question'][i]
        options = question_dataset['options'][i]

        contexts = retriever.retrieve([question])

        answer = generator.generate_answer_cfg(contexts, question, options, alpha)

        #answers_no_guidance = answers['answers']
        #answers_with_guidance = answers['guided_answers']
        
        reference_answer = question_dataset['mc_answer'][i][0]
        #print(f"Answer: {answer}")
        #print(f"Reference answer: {reference_answer}")
        if answer == reference_answer:
            correct += 1


        # results.extend([{
        #     "question"         : q,
        #     "context"          : c,
        #     "generated_answer" : a,
        #     "generated_answer_with_guidance" : ga,
        #     "reference_answer" : ra
        # } for q, c, a, ga, ra in zip(questions, contexts, answers_no_guidance, answers_with_guidance, reference_answer)])
    
    #save_to_json(results, save_path, result_type="answers with CFG")
    accuracy = correct / len(question_dataset)
    print(f"Alpha: {alpha} / Accuracy: {accuracy}")

def test_cfg_batched(question_dataset, retriever, generator, save_path, alpha,
             batch_size = 16, silent=True, max_samples=None):
    
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    
    correct = 0
    results = []

    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size} with alpha = {alpha}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        options = batch['options']

        contexts, scores = retriever.retrieve_with_score(questions)

        reference_answers = batch['mc_answer']
        answers = generator.cfg_batch(questions, contexts, options, alpha, reference_answers, scores)
        # if i == 0:
        #     print(f"Answers: {answers}")

        cfg_answers = answers['cfg_answers']
        no_context_answers = answers['no_context_answers']
        answers_with_context = answers['answers_with_context']
        alphas = answers['alphas']
        #logits_ctx = answers['logits_ctx']
        #logits_noc = answers['logits_noc']

        for cfg_answer, no_context_answer, reference_answer in zip(cfg_answers,no_context_answers,reference_answers):
            # if i == 0:
            #     print("Samples from first batch:\n"
            #         f"no_context_answer: {no_context_answer}\n"
            #         f"cfg_answer: {cfg_answer}\n"
            #         f"reference_answer: {reference_answer}\n"
            #         )
            if cfg_answer[0] == reference_answer[0]:
                correct += 1


            # Should include:
            # - cfg_answer
            # - no_context_answer
            # - context_answer
            # - reference_answer 
            # - retrieval score
            # - algebraic alpha 


        results.extend([{
            "question"         : q,
            "context"          : c,
            "answer_with_context" : ac,
            "no_context_answer" : noc,
            "cfg_answer" : cfg,
            "reference_answer" : ra,
            "algebraic_alpha": a,
            "retrieval_score" : s.tolist(),
            #"logits_ctx" : lc.tolist(),
            #"logits_noc" : ln.tolist()
        } for q, c, cfg, noc, ra, a, ac, s, in zip(questions, 
            contexts, cfg_answers, no_context_answers, reference_answers, alphas, answers_with_context, 
            scores,)]) #logits_ctx, logits_noc)])
    
    save_to_json(results, save_path, result_type="answers with CFG")
    accuracy = correct / len(question_dataset)
    print(f"Alpha: {alpha} / Accuracy: {accuracy}")