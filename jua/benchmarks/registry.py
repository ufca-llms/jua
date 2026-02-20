from __future__ import annotations

import json
import os

from jua.tasks import Task, DatasetSpec

DEFAULT_DATASET_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.json")


def _load_registry(registry_path: str) -> list[dict]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset registry must be a list of entries.")
    return data


def list_registered_datasets(registry_path: str | None = None) -> list[str]:
    path = registry_path or DEFAULT_DATASET_REGISTRY_PATH
    return [entry.get("id") for entry in _load_registry(path)]


def get_dataset_tasks(dataset_id: str, registry_path: str | None = None) -> list[Task]:
    path = registry_path or DEFAULT_DATASET_REGISTRY_PATH
    entry = next((item for item in _load_registry(path) if item.get("id") == dataset_id), None)
    if not entry:
        raise ValueError(f"Unknown dataset id: {dataset_id}. Add it to {path}.")

    source = entry.get("source", "hf")
    if source == "hf":
        dataset = DatasetSpec(
            source="hf",
            hf_id=entry.get("hf_id"),
            corpus_file=entry.get("corpus_file", "corpus.jsonl"),
            queries_file=entry.get("queries_file", "queries.jsonl"),
            qrels_file=entry.get("qrels_file", "qrels/test.tsv"),
        )
    else:
        dataset = DatasetSpec(
            source="local",
            path=entry.get("path"),
            corpus_file=entry.get("corpus_file", "corpus.jsonl"),
            queries_file=entry.get("queries_file", "queries.jsonl"),
            qrels_file=entry.get("qrels_file", "qrels/test.tsv"),
        )

    return [Task(name=entry.get("name", dataset_id), dataset=dataset)]


def get_all_tasks(registry_path: str | None = None) -> list[Task]:
    path = registry_path or DEFAULT_DATASET_REGISTRY_PATH
    tasks: list[Task] = []
    for entry in _load_registry(path):
        tasks.extend(get_dataset_tasks(entry.get("id"), registry_path=path))
    return tasks
