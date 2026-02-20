from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from jua.tasks import Task, load_task_dataset


def _safe_name(name: str) -> str:
    return name.replace("/", "_")


def _overall_score(metrics: dict[str, Any], overall_metric: str | None = None) -> float | None:
    if overall_metric:
        metric_key, at_k = overall_metric.split("@")
        metric_key = metric_key.lower()
        block = metrics.get(metric_key) or metrics.get(metric_key.upper())
        if isinstance(block, dict) and at_k:
            value = block.get(f"{metric_key.upper()}@{at_k}") or block.get(f"{metric_key}@{at_k}")
            if value is not None:
                return float(value)
        return None

    ndcg = metrics.get("ndcg") or metrics.get("NDCG")
    if isinstance(ndcg, dict):
        value = ndcg.get("NDCG@10")
        if value is not None:
            return float(value)
    _map = metrics.get("map") or metrics.get("MAP")
    if isinstance(_map, dict):
        value = _map.get("MAP@10")
        if value is not None:
            return float(value)
    return None


def run(model, tasks: list[Task], output_dir: str = "leaderboard", overall_metric: str | None = None, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for task in tasks:
        corpus, queries, qrels = load_task_dataset(task)
        dataset_name = task.name

        model_result = model.evaluate(
            corpus,
            queries,
            qrels,
            dataset_name=dataset_name,
            dataset_path=task.dataset.path,
            **kwargs,
        )

        overall = _overall_score(model_result.metrics, overall_metric=overall_metric)

        payload = {
            "model": model.name,
            "task": task.name,
            "metrics": model_result.metrics,
            "overall_score": overall,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        model_dir = os.path.join(output_dir, _safe_name(model.name))
        os.makedirs(model_dir, exist_ok=True)
        meta_file = os.path.join(model_dir, "model_meta.json")
        with open(meta_file, "w", encoding="utf-8") as meta_out:
            meta_obj = getattr(model, "meta", None)
            meta_dict = meta_obj.to_dict() if meta_obj else {}
            json.dump(
                {
                    "model": model.name,
                    "kind": getattr(model, "kind", None),
                    "meta": meta_dict,
                },
                meta_out,
                ensure_ascii=False,
                indent=2,
            )

        output_file = os.path.join(model_dir, f"{_safe_name(task.name)}.json")
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, ensure_ascii=False, indent=2)

        if model_result.results is not None:
            results_file = os.path.join(model_dir, f"{_safe_name(task.name)}_results.json")
            with open(results_file, "w", encoding="utf-8") as results_out:
                json.dump(model_result.results, results_out, ensure_ascii=False)

        results.append(payload)

    return results
