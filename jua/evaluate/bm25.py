import json
import os
from typing import Optional

import requests
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval


def _ensure_results_dir(results_file: str) -> None:
    results_dir = os.path.dirname(results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)


def _write_pyserini_jsonl(corpus: dict[str, dict[str, str]], dataset_path: str) -> str:
    os.makedirs(dataset_path, exist_ok=True)
    pyserini_jsonl = os.path.join(dataset_path, "pyserini.jsonl")
    with open(pyserini_jsonl, "w", encoding="utf-8") as f_out:
        for doc_id, doc in corpus.items():
            title, text = doc.get("title", ""), doc.get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, f_out)
            f_out.write("\n")
    return pyserini_jsonl


def _check_server(server_url: str) -> None:
    try:
        requests.get(server_url, timeout=5)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Anserini server is not reachable. Start it with:\n"
            'docker run -p 8000:8000 -e JAVA_TOOL_OPTIONS="-Xms1024m -Xmx8g" --memory=12g --memory-swap=12g -it beir/pyserini-fastapi\n'
            f"and confirm {server_url} is reachable."
        ) from exc


def evaluate_bm25(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, str]],
    dataset_path: str,
    dataset_name: str,
    results_file: str = "results/anserini_bm25.json",
    server_url: str = "http://127.0.0.1:8000",
    index_name: Optional[str] = None,
    chunk_size: int = 100,
) -> None:
    """
    Evaluate BM25 using Anserini via the pyserini-fastapi server.
    """
    _check_server(server_url)

    pyserini_jsonl = _write_pyserini_jsonl(corpus, dataset_path)

    with open(pyserini_jsonl, "rb") as f_in:
        requests.post(f"{server_url}/upload/", files={"file": f_in}, verify=False, timeout=120)

    if not index_name:
        index_name = f"beir/{dataset_name}"
    requests.get(f"{server_url}/index/", params={"index_name": index_name}, timeout=120)

    retriever = EvaluateRetrieval()
    _ensure_results_dir(results_file)

    if os.path.exists(results_file):
        results = json.load(open(results_file, "r"))
    else:
        results = {}

    print(f"{len(results)} queries already retrieved, resuming from there...")

    for i in tqdm(range(len(results), len(queries), chunk_size)):
        chunk_queries = dict(list(queries.items())[i:i + chunk_size])
        qids = list(chunk_queries.keys())
        query_texts = [chunk_queries[qid] for qid in qids]
        payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

        response = requests.post(
            f"{server_url}/lexical/batch_search/",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        results.update(response.json()["results"])
        json.dump(results, open(results_file, "w"))

    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    json.dump(results, open(results_file, "w"))

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    print(f"NDCG: {ndcg}")
    print(f"_MAP: {_map}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"MRR: {mrr}")

    metrics_file = results_file.replace(".json", "_metrics.json")
    json.dump(
        {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        },
        open(metrics_file, "w"),
    )
