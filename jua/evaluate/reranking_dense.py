from beir.retrieval.evaluation import EvaluateRetrieval
from jua.models.bm25 import CustomBM25
import json, torch, pickle
from tqdm import tqdm
from jua.models.openai_embeddings import OpenAIEmbeddings
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense.util import cos_sim, dot_score, pickle_load

def evaluate_reranking_dense(
        qrels: dict[str, dict[str, str]],
        model_name: str,
        results_path: str = "results/anserini_bm25.json",
    ):
    results_path = "results/anserini_bm25_normas-tcu.json"
    results = json.load(open(results_path, "r"))
    rerank_results = {}
    
    # encoded_path = f"./embeddings/sbert_{model_name.replace('/', '_')}"
    encoded_path = "embeddings/normas-tcu/openai_text-embedding-3-small"
    print(encoded_path)
    corpus_embeddings_0, corpus_ids_0 = pickle_load(f"{encoded_path}/corpus.0.pkl")
    queries_embeddins, query_ids = pickle_load(f"{encoded_path}/queries.pkl")
    print(len(query_ids))
    

    for query_id in tqdm(results):
        query_index = query_ids.index(query_id)
        query_embedding = queries_embeddins[query_index]

        doc_scores = {}
        for doc_id in results[query_id]:
            if doc_id in corpus_ids_0:
                doc_index = corpus_ids_0.index(doc_id)
                doc_embedding = corpus_embeddings_0[doc_index]
                score = cos_sim(query_embedding, doc_embedding).item()
                doc_scores[doc_id] = score

        # Sort documents by score
        sorted_docs = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
        rerank_results[query_id] = sorted_docs

    
    

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, [1, 3, 5, 10, 100],ignore_identical_ids=False)
    
    
    mrr = EvaluateRetrieval.evaluate_custom(qrels, rerank_results, [1, 3, 5, 10, 100], metric="mrr")

    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}, MRR: {mrr}")

    safe_model_name = model_name.replace("/", "_")
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr  
    }, open(f"results/{safe_model_name}_reranked_metrics.json", "w"))
    json.dump(rerank_results, open(f"results/{safe_model_name}_reranked_normas-tcu.json", "w")) 

    