from __future__ import annotations

import json
import os

from jua.models.api_models import (
    SbertModel,
    OpenAIEmbeddingsModel,
    GeminiEmbeddingsModel,
    BM25AnseriniModel,
    RerankDenseModel,
    RerankMonoT5Model,
)
from jua.models.custom_random import RandomModel
from jua.models.model_meta import ModelMeta


DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.json")


def _merge_meta(model, extra_meta):
    if not extra_meta:
        return model
    if isinstance(extra_meta, dict):
        extra_meta = ModelMeta.from_dict(extra_meta)
    model.meta = model.meta.merge(extra_meta)
    return model


def _load_registry(registry_path: str) -> list[dict]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Model registry must be a list of entries.")
    return data


def list_registered_models(registry_path: str | None = None) -> list[str]:
    path = registry_path or DEFAULT_REGISTRY_PATH
    return [entry.get("id") for entry in _load_registry(path)]


def get_model(model_id: str, **kwargs):
    registry_path = kwargs.get("registry_path") or DEFAULT_REGISTRY_PATH
    registry = _load_registry(registry_path)
    entry = next((item for item in registry if item.get("id") == model_id), None)
    if not entry:
        raise ValueError(f"Unknown model id: {model_id}. Add it to {registry_path}.")

    adapter = entry.get("adapter")
    if not adapter:
        raise ValueError(f"Model entry {model_id} missing 'adapter'.")

    # Adapter factory
    if adapter == "sbert":
        model = SbertModel(
            entry["model_name"],
            batch_size=kwargs.get("batch_size", entry.get("batch_size", 128)),
            devices=kwargs.get("devices") or entry.get("devices"),
        )
    elif adapter == "openai":
        model = OpenAIEmbeddingsModel(
            entry["model_name"],
            batch_size=kwargs.get("batch_size", entry.get("batch_size", 128)),
            max_tokens=kwargs.get("max_tokens", entry.get("max_tokens", 3000)),
        )
    elif adapter == "gemini":
        model = GeminiEmbeddingsModel(entry["model_name"], batch_size=kwargs.get("batch_size", entry.get("batch_size", 128)))
    elif adapter == "bm25/anserini":
        model = BM25AnseriniModel(
            index_name=kwargs.get("index_name") or entry.get("index_name"),
            server_url=kwargs.get("server_url") or entry.get("server_url", "http://127.0.0.1:8000"),
            chunk_size=kwargs.get("chunk_size") or entry.get("chunk_size", 100),
        )
    elif adapter == "rerank/dense":
        model = RerankDenseModel(
            entry["model_name"],
            results_file=kwargs.get("results_file") or entry.get("results_file"),
            embeddings_dir=kwargs.get("embeddings_dir") or entry.get("embeddings_dir"),
        )
    elif adapter == "rerank/monot5":
        model = RerankMonoT5Model(
            entry["model_name"],
            results_file=kwargs.get("results_file") or entry.get("results_file"),
            batch_size=kwargs.get("batch_size", entry.get("batch_size", 128)),
        )
    elif adapter == "random":
        model = RandomModel(
            dim=entry.get("dim", 384),
            seed=entry.get("seed", 42),
            batch_size=kwargs.get("batch_size", entry.get("batch_size", 128)),
        )
    else:
        raise ValueError(f"Unknown adapter: {adapter}")

    if entry.get("meta"):
        model = _merge_meta(model, entry["meta"])
    if kwargs.get("model_meta"):
        model = _merge_meta(model, kwargs.get("model_meta"))
    return model
