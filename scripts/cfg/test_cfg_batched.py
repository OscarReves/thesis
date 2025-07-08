from src.generator import get_generator
from src.pipeline import test_cfg_batched
from src.retriever import get_retriever
from src.utils import load_questions_by_type
from src.utils import load_knowledge_base

def main():
    kb_path = '/dtu/p1/oscrev/wiki/chunked_paragraph'
    kb_type = 'wiki'
    index_path = '/dtu/p1/oscrev/wiki/index/index_paragraph'

    documents = load_knowledge_base(kb_path, kb_type) # is now abstracted 

    print("Loading questions...")
    question_dataset = load_questions_by_type(None, type = 'citizenship', split=True)
    question_dataset = question_dataset['train']


    print("Loading retriever...")
    retriever = get_retriever(
        'e5',
        documents = documents,
        index_path = index_path,
        device = 'cuda',
        top_k = 5,
        )


    print("Loading generator...")
    generator = get_generator('gemma-9b')

    #for alpha in [-3.0, -1.0, -0.5, 0, 1.0, 3.0]:
    for alpha in [1.0]:
        test_cfg_batched(
            question_dataset=question_dataset,
            retriever=retriever,
            generator=generator,
            save_path='results/cfg/test',
            alpha=alpha,
            max_samples=None,
            batch_size=16
        )
    

if __name__ == "__main__":
    main()