from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from jua.models.openai_embeddings import OpenAIEmbeddings
from beir.retrieval.evaluation import EvaluateRetrieval
import json


def evaluate_openai_embeddings(
    model_name: str, 
    corpus: dict[str, dict[str, str]], 
    queries: dict[str, str], 
    qrels: dict[str, dict[str, str]]):

    embeddings = OpenAIEmbeddings(model_name=model_name,initialize=False)

    model = DRES(embeddings,batch_size=128)
    
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    results = retriever.encode_and_retrieve(corpus, queries)

    json.dump(results, open("results/openai_embeddings.json", "w"))

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}")
    # mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision
        # "MRR": mrr  
    }, open("results/openai_embeddings_metrcs.json", "w"))