from beir.retrieval.evaluation import EvaluateRetrieval
from jua.models.bm25 import CustomBM25
import json

def evaluate_bm25(corpus: dict[str, dict[str, str]], queries: dict[str, str], qrels: dict[str, dict[str, str]]):
    model = CustomBM25(index_path="./data/bm25_index", language="pt")
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    c = 0
    for query_id, result in results.items():
        if query_id in result.keys():
            c += 1
    json.dump(results, open("results/bm25.json", "w"))
    print(f"Number of queries with results: {c}")

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values,ignore_identical_ids=False)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    print(f"NDCG: {ndcg}")
    print(f"_MAP: {_map}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"MRR: {mrr}")

    json.dump({
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr
    }, open("results/bm25_metrics.json", "w"))

