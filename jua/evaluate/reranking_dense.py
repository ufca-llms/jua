from beir.retrieval.evaluation import EvaluateRetrieval
from jua.models.bm25 import CustomBM25
import json
from jua.models.openai_embeddings import OpenAIEmbeddings
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

def evaluate_reranking_dense(
        corpus: dict[str, dict[str, str]], 
        queries: dict[str, str], 
        qrels: dict[str, dict[str, str]],
        model_name: str,
        # model_type: str
    ):
    # _, model_type = model_name.split("_")

    model = CustomBM25(index_path="./data/bm25_index", language="pt")
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    # dense_model = OpenAIEmbeddings(model_name=model_name)
    dense_model = SentenceBERT(model_name=model_name)
    # if model_type == "openai":
    #     dense_model = OpenAIEmbeddings(model_name=model_name)
    # elif model_type == "sbert":
    #     dense_model = SentenceBERT(model_name=model_name)
    # elif model_type == "monot5":
    #     raise NotImplementedError("Monot5 is not supported for reranking")
    # else:
    #     raise ValueError(f"Invalid model type: {model_type}")

    model = DRES(
        dense_model,
        batch_size=128
    )

    dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1, 3, 5, 10, 100])

    rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

    json.dump(rerank_results, open(f"results/{model_name}_reranked.json", "w"))

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values,ignore_identical_ids=False)
    
    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}")
    # mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision
        # "MRR": mrr  
    }, open(f"results/{model_name}_reranked_metrics.json", "w"))
    json.dump(rerank_results, open(f"results/{model_name}_reranked_metrics.json", "w")) 

    