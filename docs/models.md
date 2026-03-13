# Model Registry

Models are registered in `jua/models/registry.json`. Each entry includes:
- `id`: identifier used by the CLI
- `adapter`: model type (e.g., `sbert`, `openai`, `rerank/dense`)
- `model_name`: model name (HF or similar)
- `meta`: metadata shown in the leaderboard

If you only want to use a model locally, editing your local `jua/models/registry.json` is enough.
If you want other people to be able to evaluate the model through this repository, you should open a pull request adding the model entry to the registry.

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

## Sharing a model with the community
If the goal is to make the model available for other users of this repository, the model registration should be submitted through a pull request.

The PR should include:
- the new entry in `jua/models/registry.json`
- enough metadata for leaderboard display and reproducibility
- leaderboard files under `leaderboard/<model>/` if you are also submitting benchmark results

At minimum, the model metadata should make it clear:
- what the model is
- where it comes from
- whether it is a retrieval model or reranker
- where others can find it again (for example, a Hugging Face URL)
