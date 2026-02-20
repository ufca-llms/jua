from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from jua.models.model_meta import ModelMeta


@dataclass
class ModelResult:
    metrics: Dict[str, Any]
    results: Dict[str, Dict[str, float]] | None = None


class BaseModel:
    name: str
    kind: str
    meta: ModelMeta

    def __init__(self, name: str, kind: str, meta: ModelMeta | None = None):
        self.name = name
        self.kind = kind
        self.meta = meta or ModelMeta()

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs) -> ModelResult:
        raise NotImplementedError
