from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class MultiDeviceSentenceBERT:
    def __init__(self, model_name: str, max_length: int = 3072, devices: list[str] | None = None):
        self.model_name = model_name
        self.max_length = max_length
        self.devices = devices or []
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = max_length
        self.pool = None

    def _encode(self, texts: Iterable[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=np.float32)

        if len(self.devices) > 1:
            if self.pool is None:
                self.pool = self.model.start_multi_process_pool(target_devices=self.devices)
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                pool=self.pool,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        else:
            device = self.devices[0] if self.devices else None
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                device=device,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, queries: Iterable[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        return self._encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        texts: list[str] = []
        for doc in corpus:
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            if title:
                texts.append(f"{title} {text}".strip())
            else:
                texts.append(text)
        return self._encode(texts, batch_size=batch_size, **kwargs)

    def close(self) -> None:
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None
