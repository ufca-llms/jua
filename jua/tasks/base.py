from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DatasetSpec:
    source: Literal["local", "hf"]
    path: Optional[str] = None
    hf_id: Optional[str] = None
    corpus_file: str = "corpus.jsonl"
    queries_file: str = "queries.jsonl"
    qrels_file: str = "qrels/test.tsv"


@dataclass
class Task:
    name: str
    dataset: DatasetSpec
    split: str = "test"
