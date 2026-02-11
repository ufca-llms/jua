from google import genai
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
load_dotenv()

class GeminiEmbeddings:
    def __init__(
        self,
        model_name: str = "gemini-1.5-embedding",
        batch_size: int = 128,
        max_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.client = genai.Client()
        self.batch_size = batch_size
        self.max_tokens = max_tokens    
    def truncate_text(self, text: str):
        total_tokens = self.client.models.count_tokens(
            model="gemini-2.0-flash", contents=text
        ).total_tokens
        if total_tokens > self.max_tokens:
            # Approximate truncation by characters
            avg_token_length = len(text) / total_tokens
            max_chars = int(self.max_tokens * avg_token_length)
            return text[:max_chars]
        return text

    def encode(self, texts: list[str]):
        """Encode a list of texts using Gemini embeddings API with batching."""
        if not texts:
            return []
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i:i+self.batch_size]
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,

            )
            all_embeddings.extend([embedding.values for embedding in result.embeddings])
            sleep(1)  # To avoid rate limiting
        return all_embeddings

    def encode_queries(self, queries: dict[str, str], **kwargs):
        """Encode queries - extract text values from the queries dict."""
        print("Encoding queries...")
        query_texts = queries
        encoded_queries = self.encode(query_texts)
        return encoded_queries
    
    def encode_corpus(self, corpus: dict[str, dict[str, str]], **kwargs):
        """Encode corpus - extract text values from the corpus dict."""
        print("Encoding corpus...")
        corpus_texts = [self.truncate_text(doc['text']) for doc in corpus]
        encoded_corpus = self.encode(corpus_texts)
        return encoded_corpus