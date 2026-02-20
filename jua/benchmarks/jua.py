from __future__ import annotations

from jua.tasks import Task, DatasetSpec


def get_tasks(
    source: str = "hf",
    dataset_path: str = "./jua-dataset",
    hf_id: str = "ufca-llms/jua",
) -> list[Task]:
    if source == "local":
        dataset = DatasetSpec(
            source="local",
            path=dataset_path,
            corpus_file="corpus.jsonl",
            queries_file="queries.jsonl",
            qrels_file="qrels/test.tsv",
        )
    elif source == "hf":
        dataset = DatasetSpec(
            source="hf",
            hf_id=hf_id,
            corpus_file="corpus.jsonl",
            queries_file="queries.jsonl",
            qrels_file="qrels/test.tsv",
        )
    else:
        raise ValueError(f"Unsupported source: {source}")

    return [Task(name="JuaRetrieval", dataset=dataset)]
