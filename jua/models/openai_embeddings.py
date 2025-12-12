from openai import OpenAI
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import tiktoken

load_dotenv()

class OpenAIEmbeddings: 
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = os.getenv("OPENAI_API_KEY"),
        initialize: bool = True,
        index_path: str = "./data/openai_embeddings_index.pkl",
        batch_size: int = 128,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.initialize = initialize
        self.client = OpenAI(api_key=api_key)
        self.index_path = index_path
        self.batch_size = batch_size

    def truncate_text(self, text: str, max_tokens: int = 2048):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            return encoding.decode(tokens[:max_tokens])
        return text

    def encode(self, texts: list[str]):
        """Encode a list of texts using OpenAI embeddings API with batching."""
        if not texts:
            return []
        texts = [self.truncate_text(text) for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            all_embeddings.extend([embedding.embedding for embedding in response.data])
        return all_embeddings

    def encode_queries(self, queries: dict[str, str], **kwargs):
        """Encode queries - extract text values from the queries dict."""
        print("Encoding queries...")
        query_texts = list(queries.values())
        encoded_queries = self.encode(query_texts)
        return encoded_queries
    
    def encode_corpus(self, corpus: dict[str, dict[str, str]], **kwargs):
        """Encode corpus - extract text values from the corpus dict."""
        print("Encoding corpus...")
        if self.initialize:
            # Extract text from each document in the corpus
            corpus_texts = [doc['text'] for doc in corpus]
            encoded_corpus = self.encode(corpus_texts)
            # Save as list (in same order as corpus.values())
            with open(self.index_path, "wb") as f:
                pickle.dump(encoded_corpus, f)
            return encoded_corpus
        else:
            if Path(self.index_path).exists():
                with open(self.index_path, "rb") as f:
                    encoded_corpus = pickle.load(f)
                return encoded_corpus
            else:
                print(f"Index file not found at {self.index_path}, initializing new index")
                self.initialize = True
                return self.encode_corpus(corpus, **kwargs)