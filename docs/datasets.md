# Dataset Registry

Datasets are registered in `jua/benchmarks/registry.json`.
Each entry includes:
- `id`: identifier used by the CLI
- `name`: task name (e.g., `JuaRetrieval`)
- `source`: `hf` or `local`
- `hf_id` (if `hf`)
- `path` (if `local`)
- `corpus_file`, `queries_file`, `qrels_file`

If you only want to evaluate a dataset locally, editing your local `jua/benchmarks/registry.json` is enough.
If you want other people to be able to run the dataset through this repository, you should open a pull request adding the dataset entry to the registry.

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

## Sharing a dataset with the community
If the goal is to make the dataset available to other users of this repository, the dataset registration should be submitted through a pull request.

The PR should include:
- the new entry in `jua/benchmarks/registry.json`
- enough information to locate the dataset files
- the expected task name and file layout

At minimum, the dataset registration should make it clear:
- whether the dataset is hosted on Hugging Face or stored locally
- where the corpus, queries, and qrels are located
- which split is intended for evaluation
- which task name should appear in the leaderboard
