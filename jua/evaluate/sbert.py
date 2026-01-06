from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import json

def evaluate_sbert(
    model_name: str, 
    corpus: dict[str, dict[str, str]], 
    queries: dict[str, str], 
    qrels: dict[str, dict[str, str]],
    batch_size: int):

    model_name_or_path = model_name

    dense_model = models.SentenceBERT(
        model_name_or_path
    )

    model = DRES(
        dense_model,
        batch_size=batch_size
    )
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    results = retriever.retrieve(corpus, queries)

    c = 0
    for q, r in results.items():
        if q in r:
            c += 1
    print(f"Number of queries with at least one relevant document: {c}")

    safe_model_name = model_name.replace("/", "_")
    json.dump(results, open(f"results/sbert_{safe_model_name}.json", "w"))

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
    }, open(f"results/sbert_{safe_model_name}_metrics.json", "w"))