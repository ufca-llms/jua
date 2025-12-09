import bm25s
import Stemmer
import numpy as np

class BM25Scoring:
    def __init__(self, corpus):
        self.corpus = corpus
        self.stemmer = Stemmer.Stemmer('pt')
        self.corpus_tokens = bm25s.tokenize(self.corpus, self.stemmer)
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.corpus_tokens)

    def score(self, query, iloc,k):
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = self.bm25.retrieve(query_tokens,k=k)
        # return the rank position of the iloc item
        return np.argsort(scores[0])[::-1][iloc]