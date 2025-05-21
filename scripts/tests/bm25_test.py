from src.retriever import get_retriever
from src.utils import load_documents

def main():
    documents = load_documents('/dtu/p1/oscrev/news/2025/chunked_paragraph.jsonl')
    bm25_retriever = get_retriever('bm25', index_path=None, documents=documents)
    question = "Hvem nænvnte kong Frederik i sin nytårstale?"
    results = bm25_retriever.retrieve([question])
    print(f"Retrieved contexts: {results}")


if __name__ == "__main__":
    main()