import yaml
from src.utils import load_retrieval_corpus
from src.retriever import get_retriever
from src.embedder import get_embedder
from src.generator import get_generator
from src.indexer import FaissIndexer
from src import pipeline as pipeline_module
from src.utils import save_to_json, load_documents, get_retrieval_accuracy, load_web_faq
import argparse 
from tqdm import tqdm
import os 
import numpy as np


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    documents_path = config['documents_path']
    index_path = config['index_path']
    retriever_name = config['retriever_name']
    embedder_name = config['embedder_name']
    device = config['device']
    save_path = config['save_path']
    batch_size = config['batch_size']
    
    # 1. Build index 
    documents = load_web_faq(documents_path)
    
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
        
        # indexer.index_documents_with_uid(
        #     documents=documents,
        #     batch_size=batch_size
        # )

        indexer.index_pretokenized(
            documents=documents,
            tokenized_path='data/training/tokenized_e5_inputs.pt',
            batch_size=1024
        )

        print(f"Index built and saved to {index_path}")

    # 2. Retrieve 
    if not os.path.exists(save_path):
        print(f"No retrieval results found at {save_path}, retrieving new results...")

        retriever = get_retriever(
            retriever_name,
            documents = documents,
            index_path = index_path,
            device = device,
            top_k = 100
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

            #if i // batch_size == 5:
            #    break

        save_to_json(results, save_path, result_type="retrieved uids")
    
    # 3. Evaluate 
    results = load_documents(save_path)
    results = results.filter(lambda x: x['query'] != '') # filter for missing queries 

    # for k in [1,5,10,25,50,100,1000]:
    #     accuracy = get_retrieval_accuracy(results, k = k)
    #     print(f"Retrieval accuracy@{k}: {accuracy}")
    uids = np.array(results['uid'], dtype=np.int64)
    retrieved_uids_full = np.array(results['retrieved_uids'], dtype=np.int64)

    for k in [1, 5, 10, 25, 50, 100, 100]:
        retrieved_uids = retrieved_uids_full[:, :k]
        accuracy = (uids[:, None] == retrieved_uids).any(axis=1).mean()
        print(f"Retrieval accuracy@{k}: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
