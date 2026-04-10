# jua
JUÁ — An Information Retrieval Corpus of Public Audit Court Rulings

## Paper
[JUÁ -- A Benchmark for Information Retrieval in Brazilian Legal Text Collections](https://arxiv.org/abs/2604.06098)

If you use this repository, please cite:

```bibtex
@misc{pereira2026juabenchmarkinformation,
  title={JU\'A -- A Benchmark for Information Retrieval in Brazilian Legal Text Collections},
  author={Jayr Pereira and Leandro Fernandes and Erick de Brito and Roberto Lotufo and Luiz Bonifacio},
  year={2026},
  eprint={2604.06098},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2604.06098},
}
```

## Quick Start
### Dataset generation
```
python -m jua --filepath data/jurisprudencia-selecionada.csv --directory jua-dataset
```

## New MTEB-like API
### Python API
```python
import jua

model = jua.get_model("qwen3-embedding-0.6b")
tasks = jua.get_tasks("jua", source="local", dataset_path="./jua-dataset")
jua.run(model, tasks, output_dir="results/leaderboard")
```

### CLI runner
```
python -m jua.cli \
  run \
  --model qwen3-embedding-0.6b \
  --benchmark jua \
  --output_dir results/leaderboard
```

### Leaderboard output
Results are saved under `results/leaderboard/<model>/` with one JSON per benchmark task plus `model_meta.json`.
```
results/leaderboard/
  openai_text-embedding-3-small/
    model_meta.json
    JuaRetrieval.json
```
## Model registry
Models must be registered in `jua/models/registry.json`. Each entry specifies an `id`, an `adapter`, and adapter-specific params (e.g. `model_name`, `batch_size`).

Example entry:
```json
{
  "id": "qwen3-embedding-0.6b",
  "adapter": "sbert",
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "batch_size": 128,
  "meta": {
    "name": "Qwen/Qwen3-Embedding-0.6B",
    "provider": "Qwen",
    "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    "description": "Qwen3 Embedding 0.6B",
    "model_type": ["dense-retrieval"],
    "modalities": ["text"],
    "languages": ["multilingual"]
  }
}
```

You can add extra metadata (links, authors, description) via a JSON file:
```
python -m jua.cli \
  run \
  --model qwen3-embedding-0.6b \
  --benchmark jua \
  --model_meta_json path/to/meta.json
```

### List registered models
```
python -m jua.cli list-models
```

### List registered datasets
```
python -m jua.cli list-datasets
```

The metadata file must match the `ModelMeta` schema (modeled after MTEB). Allowed fields:
`name`, `short_name`, `provider`, `description`, `url`, `authors`, `license`, `framework`, `modalities`,
`model_type`, `similarity_fn_name`, `max_tokens`, `embed_dim`, `n_parameters`,
`open_weights`, `training_data`, `training_code`, `reference`, `release_date`,
`languages`, `contacts`, `extra`.

## Adding a custom model
1) Implement an adapter class if needed (e.g., `jua/models/custom_random.py`).
2) Register the model in `jua/models/registry.json` with the adapter and params.
3) Run using the registered `id`.

## Benchmarks
### JUA (Hugging Face)
```
python -m jua.cli run --model qwen3-embedding-0.6b --benchmark jua
```
Note: HF benchmarks require `datasets` (`pip install datasets`).

### Run on all datasets
```
python -m jua.cli run --model qwen3-embedding-0.6b --all_datasets
```

## Evaluation (legacy subcommands)
### BM25 (Anserini via pyserini-fastapi)
1) Start the server:
```
docker run -p 8000:8000 -e JAVA_TOOL_OPTIONS="-Xms1024m -Xmx8g" --memory=12g --memory-swap=12g -it beir/pyserini-fastapi
```
2) Run evaluation:
```
python -m jua.evaluate bm25 --dataset_path ./data/ulysses --results_file results/anserini_bm25_ulysses.json
```

### SBERT
```
python -m jua.evaluate sbert --model_name KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 --batch_size 128 --dataset_path ./jua-dataset
```

### OpenAI/Gemini embeddings
```
python -m jua.evaluate openai --model_name text-embedding-3-small --batch_size 128 --dataset_path ./jua-dataset
```

### Reranking (dense)
```
python -m jua.evaluate rerank-dense --model_name text-embedding-3-small --results_file results/anserini_bm25_hard.json --dataset_path ./jua-dataset
```

### Reranking (MonoT5)
```
python -m jua.evaluate rerank-monot5 --model_name castorini/monot5-base-msmarco-10k --batch_size 128 --dataset_path ./jua-dataset
```

## Legacy CLI
The previous `--model_type` CLI is still supported for compatibility:
```
python -m jua.evaluate --model_type bm25 --dataset_path ./jua-dataset
```
