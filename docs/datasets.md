# Dataset Registry

Datasets are registered in `jua/benchmarks/registry.json`.
Each entry includes:
- `id`: identifier used by the CLI
- `name`: task name (e.g., `JuaRetrieval`)
- `source`: `hf` or `local`
- `hf_id` (if `hf`)
- `path` (if `local`)
- `corpus_file`, `queries_file`, `qrels_file`

## Example (HF)
```json
{
  "id": "jua",
  "name": "JuaRetrieval",
  "source": "hf",
  "hf_id": "ufca-llms/jua",
  "corpus_file": "corpus.jsonl",
  "queries_file": "queries.jsonl",
  "qrels_file": "qrels/test.tsv"
}
```

## Example (local)
```json
{
  "id": "ulysses-rfcorpus",
  "name": "UlyssesRFCorpusRetrieval",
  "source": "local",
  "path": "./data/ulysses",
  "corpus_file": "corpus.jsonl",
  "queries_file": "queries.jsonl",
  "qrels_file": "qrels/test.tsv"
}
```

## Run a dataset
```
python -m jua.cli run --model <id> --benchmark <dataset-id>
```

## Run all datasets
```
python -m jua.cli run --model <id> --all_datasets
```
