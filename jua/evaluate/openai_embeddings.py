from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from jua.models.openai_embeddings import OpenAIEmbeddings
from beir.retrieval.evaluation import EvaluateRetrieval
import json


def evaluate_openai_embeddings(
    model_name: str, 
    corpus: dict[str, dict[str, str]], 
    queries: dict[str, str], 
    qrels: dict[str, dict[str, str]],
    batch_size: int = 128,):

    embeddings = OpenAIEmbeddings(model_name=model_name,initialize=True,batch_size=batch_size,max_tokens=8192)

    model = DRES(embeddings,batch_size=batch_size)
    
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    results = retriever.encode_and_retrieve(corpus, queries, encode_output_path=f"./embeddings/openai_{model_name.replace('/', '_')}/", overwrite=False)
    
    json.dump(results, open(f"results/openai_{model_name.replace('/', '_')}.json", "w"))

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values,ignore_identical_ids=False)
    
    
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}, MRR: {mrr}")
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr  
    }, open(f"results/openai_{model_name.replace('/', '_')}_metrics.json", "w"))