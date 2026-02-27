# Model Registry

Models are registered in `jua/models/registry.json`. Each entry includes:
- `id`: identifier used by the CLI
- `adapter`: model type (e.g., `sbert`, `openai`, `rerank/dense`)
- `model_name`: model name (HF or similar)
- `meta`: metadata shown in the leaderboard

## Example (SBERT model)
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

## Example (dense reranker)
```json
{
  "id": "jua-v2-rerank",
  "adapter": "rerank/dense",
  "model_name": "ufca-llms/Qwen3-Embedding-0.6B-jua-v2",
  "meta": {
    "name": "ufca-llms/Qwen3-Embedding-0.6B-jua-v2 (rerank)",
    "provider": "ufca-llms",
    "url": "https://huggingface.co/ufca-llms/Qwen3-Embedding-0.6B-jua-v2",
    "description": "Dense reranking over BM25 results using precomputed JUA-v2 embeddings",
    "model_type": ["reranker"],
    "modalities": ["text"]
  }
}
```

## Run
```
python -m jua.cli run --model <id> --benchmark <dataset-id>
```
