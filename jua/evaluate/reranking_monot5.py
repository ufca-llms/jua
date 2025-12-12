from beir.retrieval.evaluation import EvaluateRetrieval
from jua.models.bm25 import CustomBM25
import json
from beir.reranking import Rerank
from beir.reranking.models import MonoT5

def evaluate_reranking_monot5(
        corpus: dict[str, dict[str, str]], 
        queries: dict[str, str], 
        qrels: dict[str, dict[str, str]],
        model_name: str,
        token_false: str = "_no",
        token_true: str = "_yes"
    ):
    model = CustomBM25(index_path="./data/bm25_index", language="pt")
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    cross_encoder_model = MonoT5(model_name, token_false=token_false, token_true=token_true)
    reranker = Rerank(cross_encoder_model, batch_size=128)

    rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}")
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision
    }, open(f"results/{model_name}_reranked_metrics.json", "w"))
    json.dump(rerank_results, open(f"results/{model_name}_reranked_metrics.json", "w")) 