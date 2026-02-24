from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from tqdm import tqdm

import requests
from beir.retrieval import models as beir_models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from jua.models.base import BaseModel, ModelResult
from jua.models.model_meta import ModelMeta
from jua.models.openai_embeddings import OpenAIEmbeddings
from jua.models.gemini_embeddings import GeminiEmbeddings
from jua.evaluate.reranking_dense import evaluate_reranking_dense
from jua.evaluate.reranking_monot5 import evaluate_reranking_monot5
import glob
import re


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _metrics_bundle(retriever: EvaluateRetrieval, qrels, results) -> Dict[str, Any]:
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    return {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


class SbertModel(BaseModel):
    def __init__(self, model_name: str, batch_size: int = 128):
        super().__init__(
            name=f"sbert/{model_name}",
            kind="retrieval",
            meta=ModelMeta(
                name=model_name,
                provider="sentence-transformers",
                framework=["sentence-transformers"],
                model_type=["dense-retrieval"],
                modalities=["text"],
                description="SentenceTransformer embeddings.",
                extra={"batch_size": batch_size},
            ),
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        dense_model = beir_models.SentenceBERT(
            self.model_name,
            max_length=3072,
        )
        model = DRES(dense_model, batch_size=self.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")

        embeddings_dir = kwargs.get("embeddings_dir") or f"./embeddings/{dataset_name}/sbert_{self.model_name.replace('/', '_')}"
        print(f"[sbert] Encoding corpus for dataset: {dataset_name}")
        print(f"[sbert] Encoding queries for dataset: {dataset_name}")
        print(f"[sbert] Embeddings dir: {embeddings_dir}")
        results = retriever.encode_and_retrieve(
            corpus,
            queries,
            encode_output_path=embeddings_dir,
            overwrite=kwargs.get("overwrite", False),
        )
        print(len(results), "queries retrieved", len(qrels), "queries in qrels")
        metrics = _metrics_bundle(retriever, qrels, results)
        return ModelResult(metrics=metrics, results=results)


class OpenAIEmbeddingsModel(BaseModel):
    def __init__(self, model_name: str, batch_size: int = 128, max_tokens: int = 3000):
        super().__init__(
            name=f"openai/{model_name}",
            kind="retrieval",
            meta=ModelMeta(
                name=model_name,
                provider="openai",
                model_type=["dense-retrieval"],
                modalities=["text"],
                description="OpenAI embeddings.",
                extra={"batch_size": batch_size, "max_tokens": max_tokens},
            ),
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        embeddings = OpenAIEmbeddings(
            model_name=self.model_name,
            initialize=True,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
        )
        model = DRES(embeddings, batch_size=self.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")

        embeddings_dir = kwargs.get("embeddings_dir") or f"./embeddings/{dataset_name}/openai_{self.model_name.replace('/', '_')}"
        results = retriever.encode_and_retrieve(
            corpus,
            queries,
            encode_output_path=embeddings_dir,
            overwrite=kwargs.get("overwrite", False),
        )
        metrics = _metrics_bundle(retriever, qrels, results)
        return ModelResult(metrics=metrics, results=results)


class GeminiEmbeddingsModel(BaseModel):
    def __init__(self, model_name: str, batch_size: int = 128):
        super().__init__(
            name=f"gemini/{model_name}",
            kind="retrieval",
            meta=ModelMeta(
                name=model_name,
                provider="gemini",
                model_type=["dense-retrieval"],
                modalities=["text"],
                description="Gemini embeddings.",
                extra={"batch_size": batch_size},
            ),
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        embeddings = GeminiEmbeddings(
            model_name=self.model_name,
            batch_size=self.batch_size,
        )
        model = DRES(embeddings, batch_size=self.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")

        embeddings_dir = kwargs.get("embeddings_dir") or f"./embeddings/{dataset_name}/gemini_{self.model_name.replace('/', '_')}"
        results = retriever.encode_and_retrieve(
            corpus,
            queries,
            encode_output_path=embeddings_dir,
            overwrite=kwargs.get("overwrite", False),
        )
        metrics = _metrics_bundle(retriever, qrels, results)
        return ModelResult(metrics=metrics, results=results)


class BM25AnseriniModel(BaseModel):
    def __init__(self, index_name: Optional[str] = None, server_url: str = "http://127.0.0.1:8000", chunk_size: int = 100):
        super().__init__(
            name="bm25/anserini",
            kind="retrieval",
            meta=ModelMeta(
                name="bm25/anserini",
                provider="anserini",
                model_type=["bm25"],
                modalities=["text"],
                description="BM25 via Anserini (pyserini-fastapi).",
                extra={
                    "index_name": index_name,
                    "server_url": server_url,
                    "chunk_size": chunk_size,
                },
            ),
        )
        self.index_name = index_name
        self.server_url = server_url
        self.chunk_size = chunk_size

    def _check_server(self):
        try:
            requests.get(self.server_url, timeout=5)
        except requests.RequestException as exc:
            raise RuntimeError(
                "Anserini server is not reachable. Start it with:\n"
                'docker run -p 8000:8000 -e JAVA_TOOL_OPTIONS="-Xms1024m -Xmx8g" --memory=12g --memory-swap=12g -it beir/pyserini-fastapi\n'
                f"and confirm {self.server_url} is reachable."
            ) from exc

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        self._check_server()

        dataset_path = kwargs.get("dataset_path") or "."
        _ensure_dir(dataset_path)
        pyserini_jsonl = os.path.join(dataset_path, "pyserini.jsonl")
        with open(pyserini_jsonl, "w", encoding="utf-8") as f_out:
            for doc_id, doc in corpus.items():
                title, text = doc.get("title", ""), doc.get("text", "")
                data = {"id": doc_id, "title": title, "contents": text}
                json.dump(data, f_out)
                f_out.write("\n")

        with open(pyserini_jsonl, "rb") as f_in:
            requests.post(f"{self.server_url}/upload/", files={"file": f_in}, verify=False, timeout=120)

        index_name = self.index_name or f"beir/{dataset_name}"
        requests.get(f"{self.server_url}/index/", params={"index_name": index_name}, timeout=120)

        retriever = EvaluateRetrieval()
        results = {}
        for i in tqdm(range(0, len(queries), self.chunk_size), desc="Retrieving with Anserini"):
            chunk_queries = dict(list(queries.items())[i:i + self.chunk_size])
            qids = list(chunk_queries.keys())
            query_texts = [chunk_queries[qid] for qid in qids]
            payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

            response = requests.post(
                f"{self.server_url}/lexical/batch_search/",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            results.update(response.json()["results"])

        for query_id in results:
            if query_id in results[query_id]:
                results[query_id].pop(query_id, None)

        metrics = _metrics_bundle(retriever, qrels, results)
        return ModelResult(metrics=metrics, results=results)


class RerankDenseModel(BaseModel):
    def __init__(self, model_name: str, results_file: str, embeddings_dir: Optional[str] = None):
        super().__init__(
            name=f"rerank/dense/{model_name}",
            kind="rerank",
            meta=ModelMeta(
                name=model_name,
                provider="dense",
                model_type=["reranker"],
                modalities=["text"],
                description="Dense reranker using precomputed embeddings.",
                extra={
                    "results_file": results_file,
                    "embeddings_dir": embeddings_dir,
                },
            ),
        )
        self.model_name = model_name
        self.results_file = results_file
        self.embeddings_dir = embeddings_dir

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        results_file = kwargs.get("results_file") or self.results_file or _auto_results_file(dataset_name)
        embeddings_dir = kwargs.get("embeddings_dir") or self.embeddings_dir or _auto_embeddings_dir(self.model_name, dataset_name)

        # Reuse existing pipeline (writes results to file)
        evaluate_reranking_dense(
            qrels,
            self.model_name,
            results_path=results_file,
            dataset_name=dataset_name,
            embeddings_dir=embeddings_dir,
        )
        metrics_path = f"results/{self.model_name.replace('/', '_')}_reranked_metrics.json"
        metrics = json.load(open(metrics_path, "r")) if os.path.exists(metrics_path) else {}
        return ModelResult(metrics=metrics, results=None)


class RerankMonoT5Model(BaseModel):
    def __init__(self, model_name: str, results_file: str, batch_size: int = 128):
        super().__init__(
            name=f"rerank/monot5/{model_name}",
            kind="rerank",
            meta=ModelMeta(
                name=model_name,
                provider="monot5",
                model_type=["reranker"],
                modalities=["text"],
                description="MonoT5 reranker.",
                extra={
                    "results_file": results_file,
                    "batch_size": batch_size,
                },
            ),
        )
        self.model_name = model_name
        self.results_file = results_file
        self.batch_size = batch_size

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        evaluate_reranking_monot5(
            corpus,
            queries,
            qrels,
            self.model_name,
            batch_size=self.batch_size,
        )
        metrics_path = f"results/{self.model_name.replace('/', '_')}_reranked_metrics.json"
        metrics = json.load(open(metrics_path, "r")) if os.path.exists(metrics_path) else {}
        return ModelResult(metrics=metrics, results=None)


def _slugify_benchmark(name: str) -> str:
    if name.endswith("Retrieval"):
        name = name[:-9]
    # CamelCase to kebab-case
    name = re.sub(r"([a-z0-9])([A-Z])", r"\\1-\\2", name)
    name = name.replace("_", "-").lower()
    return name


def _auto_results_file(dataset_name: str) -> str:
    slug = _slugify_benchmark(dataset_name)
    candidates = [
        f"results/anserini_bm25_{slug}.json",
        f"results/anserini_bm25_{slug.replace('-', '_')}.json",
    ]
    if slug == "jua":
        candidates.insert(0, "results/anserini_bm25_hard.json")

    for path in candidates:
        if os.path.exists(path):
            return path

    matches = [p for p in glob.glob("results/anserini_bm25_*.json") if slug in os.path.basename(p)]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Could not infer BM25 results file. Provide --results_file or set it in registry.json."
    )


def _auto_embeddings_dir(model_name: str, dataset_name: str | None = None) -> str:
    safe_model = model_name.replace("/", "_")
    candidates = [
        f"embeddings/openai_{safe_model}",
        f"embeddings/openai_{safe_model.replace('text-embedding', 'text-embedding')}",
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    matches = glob.glob(f"embeddings/**/openai_{safe_model}", recursive=True)
    if len(matches) == 1:
        return matches[0]

    # Try dataset-scoped folders (new structure)
    if dataset_name:
        ds_slug = _slugify_benchmark(dataset_name)
        dataset_candidates = [
            f"embeddings/{ds_slug}/openai_{safe_model}",
            f"embeddings/{ds_slug}/openai_{safe_model.replace('text-embedding', 'text-embedding')}",
            f"embeddings/embeddings 2/{ds_slug}",
            f"embeddings/embeddings 2/{ds_slug.replace('-', '_')}",
        ]
        for path in dataset_candidates:
            if os.path.isdir(path):
                return path

    raise FileNotFoundError(
        "Could not infer embeddings directory. Provide --embeddings_dir or set it in registry.json."
    )
