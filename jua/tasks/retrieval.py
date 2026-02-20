from __future__ import annotations

from typing import Dict, Tuple
import os

from beir.datasets.data_loader import GenericDataLoader
from .base import Task


def _load_local(task: Task) -> Tuple[Dict, Dict, Dict]:
    dataset_path = task.dataset.path or "."
    corpus_path = os.path.join(dataset_path, task.dataset.corpus_file)
    query_path = os.path.join(dataset_path, task.dataset.queries_file)
    qrels_path = os.path.join(dataset_path, task.dataset.qrels_file)

    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path,
        query_file=query_path,
        qrels_file=qrels_path,
    ).load_custom()
    return corpus, queries, qrels


def _load_hf(task: Task) -> Tuple[Dict, Dict, Dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for HF benchmarks. Install with: pip install datasets") from exc

    hf_id = task.dataset.hf_id
    if not hf_id:
        raise ValueError("HF dataset id is required for source='hf'.")

    # Support datasets that publish separate configs: corpus / queries / default (qrels)
    corpus_ds = load_dataset(hf_id, "corpus", split="corpus")
    queries_ds = load_dataset(hf_id, "queries", split="queries")
    qrels_split = "test" if task.split == "test" else task.split
    qrels_ds = load_dataset(hf_id, "default", split=qrels_split)

    corpus = {row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")} for row in corpus_ds}
    queries = {row["_id"]: row.get("text", "") for row in queries_ds}

    qrels = {}
    for row in qrels_ds:
        qid = str(row.get("query-id") or row.get("query_id") or row.get("queryId") or row.get("qid"))
        cid = str(row.get("corpus-id") or row.get("corpus_id") or row.get("corpusId") or row.get("did"))
        score = int(row.get("score") or row.get("relevance") or 1)
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = score

    return corpus, queries, qrels


def load_task_dataset(task: Task) -> Tuple[Dict, Dict, Dict]:
    if task.dataset.source == "local":
        return _load_local(task)
    if task.dataset.source == "hf":
        return _load_hf(task)
    raise ValueError(f"Unsupported dataset source: {task.dataset.source}")
