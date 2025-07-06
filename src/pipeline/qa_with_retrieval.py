import json
from tqdm import tqdm
import time
from src.utils import save_to_json

def test_qa_with_retrieval(question_dataset, retriever, generator, save_path, batch_size = 16, max_samples=None, silent=True):
    # qa pipeline meant for news questions 
    if max_samples:
        question_dataset=question_dataset.select(range(max_samples))
    results = []
    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        
        start_retrieval = time.time()
        contexts = retriever.retrieve(questions)
        total_retrieval_time = time.time() - start_retrieval
        
        start_generation = time.time()
        answers = generator.generate_batch(questions, contexts)
        total_generation_time = time.time() - start_generation
        
        reference_answers = [batch['options'][i][j] for i,j in enumerate(batch['correct_idx'])]

        results.extend([{
            "question"         : q,
            "context"          : c,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, c, a, ra in zip(questions, contexts, answers, reference_answers)])

        if not silent:
            print(f"\nRetrieval time: {total_retrieval_time:.2f}s")
            print(f"Generation time: {total_generation_time:.2f}s")
    
    save_to_json(results, save_path, result_type="answers with context")

def test_qa_with_retrieval_wiki(question_dataset, retriever, generator, save_path, 
                                batch_size=16, silent=True, max_samples=None):
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    results = []
    
    total_retrieval_time = 0.0
    total_generation_time = 0.0
    
    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        
        start_retrieval = time.time()
        contexts = retriever.retrieve_with_uid(questions)
        total_retrieval_time = time.time() - start_retrieval
        
        start_generation = time.time()
        answers = generator.generate_batch(questions, contexts)
        total_generation_time = time.time() - start_generation
        
        reference_answers = batch['answers']

        results.extend([{
            "question"         : q,
            "context"          : c,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, c, a, ra in zip(questions, contexts, answers, reference_answers)])
    
        if not silent:
            print(f"\nRetrieval time: {total_retrieval_time:.2f}s")
            print(f"Generation time: {total_generation_time:.2f}s")


    save_to_json(results, save_path, result_type="answers with context")

def test_qa_no_context(question_dataset, retriever, generator, save_path, 
                                batch_size=16, max_samples=None, silent=True):
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    results = []

    if 'answers' in question_dataset.column_names:
        answer_key = 'answers'
    else:
        answer_key = 'answer'

    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        
        answers = generator.generate_batch_no_context(questions)

        reference_answers = batch[answer_key]

        results.extend([{
            "question"         : q,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, a, ra in zip(questions, answers, reference_answers)])
    
    save_to_json(results, save_path, result_type="answers")

def test_qa_citizenship(question_dataset, retriever, generator, save_path, 
                                batch_size=16, silent=True, max_samples=None):
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    results = []
    
    total_retrieval_time = 0.0
    total_generation_time = 0.0

    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        
        start_retrieval = time.time()
        contexts = retriever.retrieve_with_uid(questions)
        total_retrieval_time = time.time() - start_retrieval
        
        start_generation = time.time()
        answers = generator.generate_batch(questions, contexts)
        total_generation_time = time.time() - start_generation
        
        reference_answer = batch['answer']

        results.extend([{
            "question"         : q,
            "context"          : c,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, c, a, ra in zip(questions, contexts, answers, reference_answer)])
        
        if not silent:
            print(f"\nRetrieval time: {total_retrieval_time:.2f}s")
            print(f"Generation time: {total_generation_time:.2f}s")
    
    save_to_json(results, save_path, result_type="answers with context")

def qa_citizenship_mc(question_dataset, retriever, generator, save_path, 
                                batch_size=16, silent=True, max_samples=None):
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    results = []
    
    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        options = batch['options']

        contexts = retriever.retrieve_with_uid(questions)

        answers = generator.generate_batch_mc(questions, contexts, options)
        
        reference_answer = batch['mc_answer']

        results.extend([{
            "question"         : q,
            "context"          : c,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, c, a, ra in zip(questions, contexts, answers, reference_answer)])
    
    save_to_json(results, save_path, result_type="answers with context")

def qa_news_mc(question_dataset, retriever, generator, save_path, 
                                batch_size=16, silent=True, max_samples=None):
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    results = []
    
    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        options = batch['options']

        contexts = retriever.retrieve(questions)

        answers = generator.generate_batch_mc(questions, contexts, options)
        
        reference_answer = batch['mc_answer']

        results.extend([{
            "question"         : q,
            "context"          : c,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, c, a, ra in zip(questions, contexts, answers, reference_answer)])
    
    save_to_json(results, save_path, result_type="answers with context")

def qa_citizenship_mc_no_context(question_dataset, retriever, generator, save_path, 
                                batch_size=16, silent=True, max_samples=None):
    if max_samples:
        question_dataset = question_dataset.select(range(max_samples))
    results = []
    
    for i in tqdm(range(0, len(question_dataset), batch_size), 
                  desc=f"Answering questions in batches of {batch_size}"):
        batch = question_dataset[i:i+batch_size]
        questions = batch['question']
        options = batch['options']

        answers = generator.generate_batch_mc_no_context(questions, options)
        
        reference_answer = batch['mc_answer']

        results.extend([{
            "question"         : q,
            "generated_answer" : a,
            "reference_answer" : ra
        } for q, a, ra in zip(questions, answers, reference_answer)])
    
    save_to_json(results, save_path, result_type="answers with context")

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

        answer = generator.generate_answer_cfg(question, contexts, options, alpha)

        #answers_no_guidance = answers['answers']
        #answers_with_guidance = answers['guided_answers']
        
        reference_answer = question_dataset['mc_answer'][i]
        if answer == reference_answer:
            correct + 1


        # results.extend([{
        #     "question"         : q,
        #     "context"          : c,
        #     "generated_answer" : a,
        #     "generated_answer_with_guidance" : ga,
        #     "reference_answer" : ra
        # } for q, c, a, ga, ra in zip(questions, contexts, answers_no_guidance, answers_with_guidance, reference_answer)])
    
    #save_to_json(results, save_path, result_type="answers with CFG")
    accuracy = correct / len(question_dataset)
    print(accuracy)
    

