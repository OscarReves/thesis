from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from datasets import load_from_disk, Dataset
import os
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
from rank_bm25 import BM25Okapi
import re
from gensim.summarization.bm25 import BM25
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity


class E5Retriever:
    def __init__(self, index_path, documents, device=None, text_field='text'):
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.device = torch.device(device)

        print(f"Loaded model {model_name} on device {self.device}")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        self.index = faiss.read_index(index_path)


        # Load dataset and extract text field
        self.dataset = documents  # or load_dataset(...)
        self.contexts = self.dataset[text_field]      # list of texts
        self.titles = self.dataset['id']
        if "uid" in self.dataset.column_names:
            self.uid_map = {int(row["uid"]): row for row in self.dataset} # construct uid mapping

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
            return F.normalize(pooled, p=2, dim=1).cpu().numpy()
    
    def retrieve(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        results = []
        for idxs in I:
            subset = self.dataset.select(idxs)
            contexts = subset["text"]
            results.append(contexts)
        
        return results
    
    def retrieve_with_uid(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        
        results = []
        for uids in I:
            subset = self.select_by_uids(uids)
            contexts = subset["text"] 
            results.append(contexts)  
        
        return results
    
    def retrieve_titles(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        return [
            # consider returning a list instead and joining somewhere else
            # likewise, consider mapping the index to documents with an ID
            [self.titles[idx] for idx in indices]
            for indices in I
        ]

    def retrieve_titles_with_uid(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        
        results = []
        for uids in I:
            subset = self.select_by_uids(uids)
            titles = subset["id"]  # grab list of titles
            results.append(titles)  # join actual strings
        
        return results

    def retrieve_uids(self, questions, top_k=5):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, top_k)
        
        return I.tolist()

    def select_by_uids(self, uids):
        rows = [self.uid_map[int(uid)] for uid in uids if int(uid) in self.uid_map]
        return Dataset.from_list(rows)


class BM25Retriever():
    def __init__(self, index_path, documents, device=None, text_field='text'):
        self.dataset = documents  
        self.contexts = self.dataset[text_field]      
        self.titles = self.dataset['id']

        # process documents
        print(f"Preprocessing {len(self.contexts)} chunks...")
        self.tokenized_contexts = [self.preprocess(doc) for doc in self.contexts]

        # build bm25 index spontaneously?
        print(f"Building bm25 index...")
        self.bm25 = BM25Okapi(self.tokenized_contexts)

    def retrieve(self, questions, top_k = 5):
        results = [self.bm25.get_top_n(
            self.preprocess(question), self.contexts, n=top_k
            ) 
            for question in questions]
        return results

    def retrieve_with_uid(self, questions, top_k = 5):
        # this is bad form, but if it works it stays 
        results = [self.bm25.get_top_n(
            self.preprocess(question), self.contexts, n=top_k
            ) 
            for question in questions]
        return results


    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = text.split()
        return tokens

class GensimBM25Retriever():
    def __init__(self, index_path, documents, device=None, text_field='text'):
        self.dataset = documents  
        self.contexts = self.dataset[text_field]      
        self.titles = self.dataset['id']

        # process documents (lowercase and remove punctuation)
        print(f"Preprocessing {len(self.contexts)} chunks...")
        self.processed_contexts = [self.preprocess(doc) for doc in self.contexts]

        # Build the bm25 index
        print(f"Building bm25 index with sparse matrices...")
        self.bm25 = BM25(self.processed_contexts) # bm25 model
        self.dictionary = Dictionary(self.processed_contexts) # define dictionary for bag-of-words conversion
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_contexts] # convert corpus to bow

        # build index
        self.index = SparseMatrixSimilarity(self.corpus, num_features=len(self.dictionary))

    def get_top_n(self, query_bow, n = 5):
        scores = self.index[query_bow]
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        return top_n 

    def retrieve(self, questions, top_k = 5):
        results = [self.get_top_n(
            self.preprocess(question), self.contexts, n=top_k
            ) 
            for question in questions]
        return results

    def retrieve_with_uid(self, questions, top_k = 5):
        # this is bad form, but if it works it stays 
        results = [self.get_top_n(
            self.preprocess(question), self.contexts, n=top_k
            ) 
            for question in questions]
        return results

    def process_query(self, query):
        lowercased = self.preprocessed(query)
        bow = self.dictionary.doc2bow(lowercased)
        return bow

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = text.split()
        return tokens

class DummyRetriever():
    def __init__(self, index_path, documents, device=None, text_field='text'):
        print("I am a stupid retriever and I don't initialize anything")
        print("If you get a segfault when running my only method the issue has to do with the global state")

    def uid_sanity_test(self):
        import numpy as np
        index = faiss.read_index("data/wiki/index/index.faiss")
        query = np.random.rand(1, 128).astype("float32")
        faiss.normalize_L2(query)
        D, I = index.search(query, 5)
        print(I)

