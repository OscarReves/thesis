from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from datasets import load_from_disk, Dataset
import os
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
from rank_bm25 import BM25Okapi
import re
# from gensim.summarization.bm25 import BM25
# from gensim.corpora import Dictionary
# from gensim.similarities import SparseMatrixSimilarity
import numpy as np
import pickle
import scipy.sparse as sp

class E5Retriever:
    def __init__(self, index_path, documents, device=None, text_field='text', top_k = 5):
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.device = torch.device(device)
        self.top_k = top_k

        print(f"Loaded model {model_name} for top-{top_k} on device {self.device}")

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
        if "id" in self.dataset.column_names:
            self.titles = self.dataset['id']
        if "uid" in self.dataset.column_names:
            self.uid_map = {int(row["uid"]): row for row in self.dataset} # construct uid mapping

    def set_top_k(self, top_k):
        self.top_k = top_k


    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
            return F.normalize(pooled, p=2, dim=1).cpu().numpy() # you are normalizing twice. Not harmful, but also not necessary 
    
    def retrieve(self, questions):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, self.top_k)
        results = []
        for idxs in I:
            subset = self.dataset.select(idxs)
            contexts = subset["text"]
            results.append(contexts)
        
        return results
    
    def retrieve_with_uid(self, questions):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, self.top_k)
        
        results = []
        for uids in I:
            subset = self.select_by_uids(uids)
            contexts = subset["text"] 
            results.append(contexts)  
        
        return results
    
    def retrieve_titles(self, questions):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, self.top_k)
        return [
            # consider returning a list instead and joining somewhere else
            # likewise, consider mapping the index to documents with an ID
            [self.titles[idx] for idx in indices]
            for indices in I
        ]

    def retrieve_titles_with_uid(self, questions):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, self.top_k)
        
        results = []
        for uids in I:
            subset = self.select_by_uids(uids)
            titles = subset["id"]  # grab list of titles
            results.append(titles)  # join actual strings
        
        return results

    def retrieve_uids(self, questions):
        queries = [f"query: {q}" for q in questions]
        q_embs = self.embed(queries)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, self.top_k)
        
        return I.tolist()

    def select_by_uids(self, uids):
        rows = [self.uid_map[int(uid)] for uid in uids if int(uid) in self.uid_map]
        return Dataset.from_list(rows)


class E5RetrieverGPU(E5Retriever):
    # Has several benefits over the parent class:
    #   1. Moves index to gpu for faster search
    #   2. Recieves an embed to allow for modular encoding of the queries 
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder 
        index_cpu = faiss.read_index(self.index)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu) # move the index to gpu

    def retrieve_uids(self, questions):
        q_embs = self.embedder.encode_query(questions)
        faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, self.top_k)
        
        return I.tolist()

# class E5RetrieverGPU:
#     def __init__(self, index_path, documents, device=None, text_field='text', top_k = 5):
#         model_name = 'intfloat/multilingual-e5-large-instruct'
#         self.device = torch.device(device)
#         self.top_k = top_k

#         print(f"Loaded model {model_name} for top-{top_k} on device {self.device}")

#         # Load model
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         if self.tokenizer.pad_token == None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)

#         # Load FAISS index
#         if not os.path.exists(index_path):
#             raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        
#         index_cpu = faiss.read_index(index_path)
#         res = faiss.StandardGpuResources()
#         self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu) # move the index to gpu 


#         # Load dataset and extract text field
#         self.dataset = documents  # or load_dataset(...)
#         self.contexts = self.dataset[text_field]      # list of texts
#         if "id" in self.dataset.column_names:
#             self.titles = self.dataset['id']
#         if "uid" in self.dataset.column_names:
#             self.uid_map = {int(row["uid"]): row for row in self.dataset} # construct uid mapping

#     def set_top_k(self, top_k):
#         self.top_k = top_k


#     def embed(self, texts):
#         inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             output = self.model(**inputs)
#             mask = inputs["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
#             pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
#             return F.normalize(pooled, p=2, dim=1).cpu().numpy() # you are normalizing twice. Not harmful, but also not necessary 
    
#     def retrieve(self, questions):
#         queries = [f"query: {q}" for q in questions]
#         q_embs = self.embed(queries)
#         faiss.normalize_L2(q_embs)
#         D, I = self.index.search(q_embs, self.top_k)
#         results = []
#         for idxs in I:
#             subset = self.dataset.select(idxs)
#             contexts = subset["text"]
#             results.append(contexts)
        
#         return results
    
#     def retrieve_with_uid(self, questions):
#         queries = [f"query: {q}" for q in questions]
#         q_embs = self.embed(queries)
#         faiss.normalize_L2(q_embs)
#         D, I = self.index.search(q_embs, self.top_k)
        
#         results = []
#         for uids in I:
#             subset = self.select_by_uids(uids)
#             contexts = subset["text"] 
#             results.append(contexts)  
        
#         return results
    
#     def retrieve_titles(self, questions):
#         queries = [f"query: {q}" for q in questions]
#         q_embs = self.embed(queries)
#         faiss.normalize_L2(q_embs)
#         D, I = self.index.search(q_embs, self.top_k)
#         return [
#             # consider returning a list instead and joining somewhere else
#             # likewise, consider mapping the index to documents with an ID
#             [self.titles[idx] for idx in indices]
#             for indices in I
#         ]

#     def retrieve_titles_with_uid(self, questions):
#         queries = [f"query: {q}" for q in questions]
#         q_embs = self.embed(queries)
#         faiss.normalize_L2(q_embs)
#         D, I = self.index.search(q_embs, self.top_k)
        
#         results = []
#         for uids in I:
#             subset = self.select_by_uids(uids)
#             titles = subset["id"]  # grab list of titles
#             results.append(titles)  # join actual strings
        
#         return results

#     def retrieve_uids(self, questions):
#         queries = [f"query: {q}" for q in questions]
#         q_embs = self.embed(queries)
#         faiss.normalize_L2(q_embs)
#         D, I = self.index.search(q_embs, self.top_k)
        
#         return I.tolist()

#     def select_by_uids(self, uids):
#         rows = [self.uid_map[int(uid)] for uid in uids if int(uid) in self.uid_map]
#         return Dataset.from_list(rows)

class BM25Retriever():
    def __init__(self, index_path, documents, device=None, text_field='text', top_k = 5):
        self.dataset = documents  
        self.contexts = self.dataset[text_field]      
        self.titles = self.dataset['id']
        self.index_path = index_path
        self.top_k = top_k

        # process documents
        print(f"Preprocessing {len(self.contexts)} chunks...")
        self.tokenized_contexts = [self.preprocess(doc) for doc in self.contexts]


        if os.path.exists(index_path):
            self.load()
        else:
            print(f"Building bm25 index...")
            self.bm25 = BM25Okapi(self.tokenized_contexts)
            self.save()
    
    def retrieve(self, questions):
        top_k = self.top_k
        results = [self.bm25.get_top_n(
            self.preprocess(question), self.contexts, n=top_k
            ) 
            for question in questions]
        return results

    def get_top_n(self, query, n = 5):
        # use numpy instead of built-in top_n method 
        scores = self.bm25.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:n]
        top_docs = [self.contexts[i] for i in top_indices]
        return top_docs

    def retrieve_with_uid(self, questions):
        # this is bad form, but if it works it stays 
        results = [self.get_top_n(
            self.preprocess(question), n=self.top_k
            ) 
            for question in questions]
        return results

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = text.split()
        return tokens

    def save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)

class SparseBM25Retriever():
    def __init__(self, index_path, documents, device=None, text_field='text', top_k = 5):
        self.dataset = documents  
        self.contexts = self.dataset[text_field]      
        self.titles = self.dataset['id']
        self.index_path = index_path
        self.top_k = top_k

        # process documents
        print(f"Preprocessing {len(self.contexts)} chunks...")
        self.tokenized_contexts = [self.preprocess(doc) for doc in self.contexts]

        if os.path.exists(index_path):
            print(f"Loading bm25 index from {index_path}...")
            self.load()
        else:
            print(f"Building bm25 index...")
            self.bm25 = BM25Okapi(self.tokenized_contexts)
            self.save()

        if os.path.exists(self.index_path + ".npz") and os.path.exists(self.index_path + ".vocab.pkl"):
            print(f"Loading sparse matrix from {self.index_path}.npz...")
            self.load_sparse_matrix()
        else:
            print("Building sparse matrix...")
            self.bm25_matrix, self.vocab = self.bm25_to_sparse_matrix(self.bm25)
            self.save_sparse_matrix()


    def retrieve(self, questions):
        results = [self.bm25.get_top_n(
            self.preprocess(question), self.contexts, n=self.top_k
            ) 
            for question in questions]
        return results

    def retrieve_with_uid(self, questions):
        # this is bad form, but if it works it stays 
        # results = [self.search_sparse_bm25(question, top_k=top_k) for question in questions]
        results = self.search_batch(questions, top_k=self.top_k)
        
        return results

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = text.split()
        return tokens

    def save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)


    def bm25_to_sparse_matrix(self,bm25):
        vocab = {term: i for i, term in enumerate(bm25.idf.keys())}
        N = len(self.tokenized_contexts)
        V = len(vocab)

        rows, cols, data = [], [], []

        for doc_id, doc in enumerate(self.tokenized_contexts):
            tf = bm25.doc_freqs[doc_id]
            doc_len = bm25.doc_len[doc_id]

            for term, freq in tf.items():
                if term not in vocab:
                    continue

                idf = bm25.idf[term]
                score = idf * ((freq * (bm25.k1 + 1)) /
                            (freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)))
                rows.append(doc_id)
                cols.append(vocab[term])
                data.append(score)

        bm25_matrix = sp.csr_matrix((data, (rows, cols)), shape=(N, V))
        return bm25_matrix, vocab
        
    def make_query_vector(self, query_tokens):
        indices = [self.vocab[t] for t in query_tokens if t in self.vocab]
        data = [1.0] * len(indices)  # Query as simple 1-hot vector
        query_vec = sp.csr_matrix((data, ([0]*len(indices), indices)), shape=(1, len(self.vocab)))
        return query_vec

    def search_sparse_bm25(self, query):
        query_tokens = self.preprocess(query)
        qvec = self.make_query_vector(query_tokens)

        # Compute BM25 relevance scores: dot product with sparse matrix
        scores = qvec @ self.bm25_matrix.T  # shape: (1, num_docs)
        scores = scores.toarray().ravel()

        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [self.contexts[i] for i in top_indices if scores[i] > 0]
    
    def search_batch(self, queries):
        query_vecs = sp.vstack([
            self.make_query_vector(self.preprocess(q)) for q in queries
        ])
        scores = query_vecs @ self.bm25_matrix.T  # shape: (num_queries, num_docs)
        results = []

        for row in scores:
            row = row.toarray().ravel()
            top_indices = np.argsort(row)[::-1][:self.top_k]
            results.append([self.contexts[i] for i in top_indices if row[i] > 0])

        return results

    def save_sparse_matrix(self):
        sp.save_npz(self.index_path + ".npz", self.bm25_matrix)
        with open(self.index_path + ".vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def load_sparse_matrix(self):
        self.bm25_matrix = sp.load_npz(self.index_path + ".npz")
        with open(self.index_path + ".vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)




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

