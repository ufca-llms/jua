"""Run the local BM25 baseline: myenv/bin/python tests/run_bm25_scifact.py."""
import json
from pathlib import Path

import bm25s
import Stemmer
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    cache = str(root / "data/hf-cache")
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus", cache_dir=cache)
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries", cache_dir=cache)
    qrels_ds = load_dataset("mteb/scifact", "default", split="test", cache_dir=cache)
    qrels = {}
    for row in qrels_ds:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = {str(row["_id"]): row["text"] for row in queries_ds if str(row["_id"]) in qrels}
    doc_ids = [str(row["_id"]) for row in corpus_ds]
    texts = [f'{row.get("title", "")} {row["text"]}'.strip() for row in corpus_ds]
    stemmer = Stemmer.Stemmer("english")
    model = bm25s.BM25(method="lucene", k1=0.9, b=0.4)
    model.index(bm25s.tokenize(texts, stopwords="en", stemmer=stemmer))
    ids, scores = model.retrieve(
        bm25s.tokenize(list(queries.values()), stopwords="en", stemmer=stemmer),
        k=min(1000, len(doc_ids)),
    )
    results = {
        qid: {doc_ids[int(idx)]: float(score) for idx, score in zip(indices, values)}
        for qid, indices, values in zip(queries, ids, scores)
    }
    evaluator = EvaluateRetrieval()
    ndcg, map_scores, recall, precision = evaluator.evaluate(
        qrels, results, evaluator.k_values, ignore_identical_ids=False
    )
    mrr = evaluator.evaluate_custom(qrels, results, evaluator.k_values, metric="mrr")
    metrics = dict(ndcg=ndcg, map=map_scores, recall=recall, precision=precision, mrr=mrr)
    output = root / "results/bm25_scifact"
    output.mkdir(parents=True, exist_ok=True)
    (output / "rankings.json").write_text(json.dumps(results), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({
        "dataset": "mteb/scifact", "split": "test", "documents": len(doc_ids),
        "queries": len(queries), "backend": "bm25s", "method": "lucene",
        "k1": 0.9, "b": 0.4, "stemmer": "english", "stopwords": "en",
        "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
