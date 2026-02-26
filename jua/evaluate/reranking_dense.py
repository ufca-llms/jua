from beir.retrieval.evaluation import EvaluateRetrieval
import json
from tqdm import tqdm
import glob
from itertools import islice
from beir.retrieval.search.dense.util import cos_sim, pickle_load


def evaluate_reranking_dense(
        qrels: dict[str, dict[str, str]],
        model_name: str,
        results_path: str = "results/anserini_bm25.json",
        dataset_name: str = "ulysses",
        embeddings_dir: str | None = None,
    ):

    results = json.load(open(results_path, "r"))
    rerank_results = {}

    model_type = "sbert"
    if model_name.startswith("text-embedding"):
        model_type = "openai"

    # encoded_path = f"./embeddings/normas-tcu/sbert_{model_name.replace('/', '_')}"
    if embeddings_dir:
        encoded_path = embeddings_dir
    else:
        encoded_path = f"./embeddings/{dataset_name}/{model_type}_{model_name.replace('/', '_')}"
    print(encoded_path)

    corpus_embeddings_files = glob.glob(f"{encoded_path}/corpus.*.pkl")
    if not corpus_embeddings_files:
        raise FileNotFoundError(f"No corpus embeddings found under {encoded_path}")

    # Build corpus id -> embedding map across all shards
    corpus_id_to_embedding = {}
    shards = map(pickle_load, corpus_embeddings_files)
    if len(corpus_embeddings_files) > 1:
        shards = tqdm(shards, desc="Loading shards into index", total=len(corpus_embeddings_files))
    for corpus_embeddings, corpus_ids in shards:
        for doc_id, emb in zip(corpus_ids, corpus_embeddings):
            corpus_id_to_embedding[doc_id] = emb

    # Build query id -> embedding map
    queries_embeddings, query_ids = pickle_load(f"{encoded_path}/queries.pkl")
    query_id_to_embedding = {qid: emb for qid, emb in zip(query_ids, queries_embeddings)}
    print(len(query_ids))

    missing_queries = 0
    missing_docs = 0
    missing_query_examples = []
    missing_doc_examples = []
    for query_id in tqdm(results):
        query_embedding = query_id_to_embedding.get(query_id)
        if query_embedding is None:
            missing_queries += 1
            if len(missing_query_examples) < 5:
                missing_query_examples.append(query_id)
            continue

        doc_scores = {}
        for doc_id in results[query_id]:
            doc_embedding = corpus_id_to_embedding.get(doc_id)
            if doc_embedding is None:
                missing_docs += 1
                if len(missing_doc_examples) < 5:
                    missing_doc_examples.append(doc_id)
                continue
            score = cos_sim(query_embedding, doc_embedding).item()
            doc_scores[doc_id] = score

        # Sort documents by score
        sorted_docs = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
        rerank_results[query_id] = sorted_docs

    print(f"Missing query embeddings: {missing_queries}")
    print(f"Missing doc embeddings: {missing_docs}")
    if missing_query_examples:
        print(f"Example queries missing embeddings: {missing_query_examples}")
    if missing_doc_examples:
        print(f"Example docs missing embeddings: {missing_doc_examples}")

    # Additional diagnostics for ID mismatches
    result_query_ids = set(results.keys())
    embedding_query_ids = set(query_id_to_embedding.keys())
    missing_query_ids = list(islice(result_query_ids - embedding_query_ids, 5))
    extra_query_ids = list(islice(embedding_query_ids - result_query_ids, 5))
    if missing_query_ids:
        print(f"Queries in results but not in embeddings (sample): {missing_query_ids}")
    if extra_query_ids:
        print(f"Queries in embeddings but not in results (sample): {extra_query_ids}")

    # For docs, show a small sample from embeddings for sanity checking
    embedding_doc_ids_sample = list(islice(corpus_id_to_embedding.keys(), 5))
    if embedding_doc_ids_sample:
        print(f"Sample doc IDs from embeddings: {embedding_doc_ids_sample}")

    # Filter to qrels keys to avoid empty evaluation or KeyErrors
    filtered_results = {qid: res for qid, res in rerank_results.items() if qid in qrels}
    if not filtered_results:
        raise ValueError("No reranked results match qrels. Check embeddings/query IDs and dataset alignment.")
    print(len(qrels))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, filtered_results, [1, 3, 5, 10, 100], ignore_identical_ids=False)

    mrr = EvaluateRetrieval.evaluate_custom(qrels, filtered_results, [1, 3, 5, 10, 100], metric="mrr")

    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}, MRR: {mrr}")

    safe_model_name = model_name.replace("/", "_")
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr
    }, open(f"results/{safe_model_name}_reranked_metrics.json", "w"))
    json.dump(filtered_results, open(f"results/{safe_model_name}_reranked_jua.json", "w")) 

