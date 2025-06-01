from src.utils import load_knowledge_base

if __name__ == "__main__":
    kb = load_knowledge_base('/dtu/p1/oscrev/wiki/chunked_paragraph',type='wiki')
    
    document_count = len(kb)
    sum = 0
    for chunk in kb:
        sum += len(chunk['text'].split())

    mean = sum/document_count

    print(f"The knowledge base contain {document_count} chunks")
    print(f"Average length: {mean}")
