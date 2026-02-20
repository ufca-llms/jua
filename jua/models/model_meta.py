from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


_ALLOWED_FIELDS = {
    "name",
    "provider",
    "description",
    "url",
    "authors",
    "license",
    "framework",
    "modalities",
    "model_type",
    "similarity_fn_name",
    "max_tokens",
    "embed_dim",
    "n_parameters",
    "open_weights",
    "training_data",
    "training_code",
    "reference",
    "release_date",
    "languages",
    "contacts",
    "extra",
}


def _validate_keys(data: Dict[str, Any]) -> None:
    unknown = set(data.keys()) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unknown model meta fields: {sorted(unknown)}")


@dataclass
class ModelMeta:
    name: Optional[str] = None
    provider: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    authors: Optional[List[str]] = None
    license: Optional[str] = None
    framework: Optional[List[str]] = None
    modalities: Optional[List[str]] = None
    model_type: Optional[List[str]] = None
    similarity_fn_name: Optional[str] = None
    max_tokens: Optional[int] = None
    embed_dim: Optional[int] = None
    n_parameters: Optional[int] = None
    open_weights: Optional[bool] = None
    training_data: Optional[str] = None
    training_code: Optional[str] = None
    reference: Optional[str] = None
    release_date: Optional[str] = None
    languages: Optional[List[str]] = None
    contacts: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMeta":
        _validate_keys(data)
        return cls(**data)

    def merge(self, other: "ModelMeta") -> "ModelMeta":
        base = asdict(self)
        other_dict = asdict(other)
        for k, v in other_dict.items():
            if v is not None and k != "extra":
                base[k] = v
        base_extra = base.get("extra") or {}
        other_extra = other_dict.get("extra") or {}
        base["extra"] = {**base_extra, **other_extra}
        return ModelMeta.from_dict(base)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Drop empty fields to keep JSON compact
        return {k: v for k, v in data.items() if v is not None and v != {}}
