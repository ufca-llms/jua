from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import json

def evaluate_sbert(
    model_name: str, 
    corpus: dict[str, dict[str, str]], 
    queries: dict[str, str], 
    qrels: dict[str, dict[str, str]]):

    model_name_or_path = model_name

    dense_model = models.SentenceBERT(
        model_name_or_path
    )

    model = DRES(
        dense_model,
        batch_size=128
    )
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    results = retriever.retrieve(corpus, queries)
    json.dump(results, open(f"results/sbert_{model_name}.json", "w"))

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
    }, open(f"results/sbert_{model_name}_metrics.json", "w"))