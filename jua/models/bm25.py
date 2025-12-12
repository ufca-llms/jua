import bm25s
import Stemmer
import numpy as np
import os
from beir.retrieval.search import BaseSearch
from tqdm import tqdm

class CustomBM25(BaseSearch):
    def __init__(self, index_path:str, language: str, initialize: bool = True, batch_size: int = 128):
        self.index_path = index_path
        self.language = language
        self.initialize = initialize
        self.stemmer = Stemmer.Stemmer(language)
        self.bm25 = bm25s.BM25()
        self.batch_size = batch_size
        
    def load_index(self):
        if os.path.exists(self.index_path):
            self.bm25.load(self.index_path)
        else:
            self.bm25 = bm25s.BM25()

    def index(self, corpus:  dict[str, dict[str, str]]):
        # corpus is a dictionary with _id, and text
        self.corpus = corpus
        self.text_corpus = [doc['text'] for doc in corpus.values()]
        self.tokens_corpus = bm25s.tokenize(self.text_corpus, self.stemmer)
        self.bm25.index(self.tokens_corpus)

        self.bm25.save(self.index_path, corpus = self.corpus)
    
    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ):
        if self.initialize:
            self.index(corpus)
        else:
            self.load_index()

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        results = {}
        for start_idx in tqdm(range(0, len(query_ids), self.batch_size), desc="Searching queries"):
            query_ids_batch = query_ids[start_idx:start_idx + self.batch_size]
            queries_batch = query_texts[start_idx:start_idx + self.batch_size]
            tokenized_queries_batch = bm25s.tokenize(queries_batch, self.stemmer)
            results_batch, scores_batch = self.bm25.retrieve(tokenized_queries_batch, k=1000)
                       
            for i, query_id in enumerate(query_ids_batch):
                doc_ids = list(self.corpus.keys())
                # results_batch[i] contains document indices, scores_batch[i] contains scores
                query_results = {}
                for j, doc_idx in enumerate(results_batch[i]):
                    if doc_idx < len(doc_ids):
                        score = scores_batch[i][j]
                        # Handle numpy array scalar conversion
                        if isinstance(score, np.ndarray):
                            score = score.item()
                        query_results[doc_ids[doc_idx]] = float(score)
                results[query_id] = query_results
        
        return results
    
    def encode(self, texts, **kwargs):
        """Encode texts for retrieval. For BM25, this returns tokenized texts."""
        if isinstance(texts, str):
            texts = [texts]
        return bm25s.tokenize(texts, self.stemmer, return_ids=False)
    
    def search_from_files(
        self,
        corpus_file: str,
        queries_file: str,
        output_file: str,
        top_k: int,
        *args,
        **kwargs,
    ):
        """Search from files. Not implemented for BM25 - use search() method instead."""
        raise NotImplementedError(
            "search_from_files is not implemented for CustomBM25. "
            "Please use the search() method with corpus and queries dictionaries."
        )
