import yaml
from src.utils import load_retrieval_corpus
from src.retriever import get_retriever
from src.embedder import get_embedder
from src.generator import get_generator
from src.indexer import FaissIndexer
from src import pipeline as pipeline_module
from src.utils import save_to_json
import argparse 
from tqdm import tqdm
import os 

def main():
    # Start by writing for one specific dataset, then generalize 
    documents = load_retrieval_corpus()
    index_path = 'results/retrieval_test/index.faiss'
    embedder_name = 'e5'
    device = 'cuda'
    retriever_name = 'e5'
    save_path = 'results/retrieval_test/results'
    batch_size = 1024

    # 1. Build index 
    embedder = get_embedder(embedder_name)

    if os.path.exists(index_path):
        print(f"Index already exists at {index_path}")
        pass
    else:
        print(f"No index found at {index_path}, building index...")
        indexer = FaissIndexer(
            embedder= embedder,
            index_path= index_path
        )
        
        indexer.index_documents_with_uid(
            documents=documents,
            batch_size=batch_size
        )

        print(f"Index built and saved to {index_path}")

    # 2. Retrieve 
    retriever = get_retriever(
        retriever_name,
        documents = documents,
        index_path = index_path,
        device = device
        )

    results = []
    for i in tqdm(range(0, len(documents), batch_size), 
                  desc=f"Retrieving documents in batches of {batch_size}"):
        batch = documents[i:i+batch_size]
        queries = batch['query']
        uids = batch['uid']

        retrieved_uids = retriever.retrieve_uids(queries) 
            # really you should pre-compute embeddings and search manually
        
        results.extend({
            "query"     : q,
            "uid"       : u,
            "retrieved_uids"    : ru
        } for q, u, ru in zip(queries, uids, retrieved_uids))

    save_to_json(results, save_path, result_type="answers with context")
    
    # 3. Evaluate 
    def uid_match(sample):
        return sample('uid') in sample('retrieved_uids')
    
    correct = results.filter(uid_match)
    accuracy = len(correct)/len(results)
    print(f"Retrieval accuracy: {accuracy}")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", required=True, help="Path to config YAML")
    #args = parser.parse_args()
    #main(args.config)
    main()