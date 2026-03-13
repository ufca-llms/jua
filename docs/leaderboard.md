# Leaderboard Outputs

## Where results are stored
- **Metrics**: `leaderboard/<model>/<Benchmark>.json`
- **Raw results**: `results/leaderboard/<model>/<Benchmark>_results.json`
- **Model metadata**: `leaderboard/<model>/model_meta.json`

## Metrics file format
```json
{
  "model": "bm25/anserini",
  "task": "JuaRetrieval",
  "metrics": {
    "ndcg": {"NDCG@10": 0.1234},
    "map": {"MAP@10": 0.0567},
    "recall": {"Recall@10": 0.3456},
    "precision": {"P@10": 0.1111},
    "mrr": {"MRR@10": 0.0789}
  },
  "overall_score": 0.1234
}
```

## Update the leaderboard
Run the model on the desired benchmark:
```
python -m jua.cli run --model <id> --benchmark <dataset-id>
```

This command updates the leaderboard files under `leaderboard/`. Raw retrieval outputs should stay under `results/`.

## Submitting a new run
Leaderboard updates should be submitted through a pull request.

### What should be included in the PR
- The metrics files under `leaderboard/<model>/JuaRetrieval.json`
- `leaderboard/<model>/model_meta.json`
- Any registry changes required to identify the model or benchmark:
  - `jua/models/registry.json`
  - `jua/benchmarks/registry.json`

### What should not be included in the PR
- Raw run files such as `*_results.json`
- Large embedding artifacts
- Temporary notebooks or local experiment files

Raw outputs belong in `results/` for local inspection, but they should generally not be committed as part of a leaderboard submission.

## Recommended PR checklist
Before opening the PR, make sure that:
- The model has a stable identifier in `jua/models/registry.json`
- The benchmark is already registered, or the PR includes the benchmark registration
- `leaderboard/<model>/model_meta.json` contains at least:
  - model name
  - provider
  - description
  - URL when available
  - `model_type`
- Each metrics file follows the expected JSON structure
- The reported benchmark names match the registered task names exactly
- The leaderboard app can read the new files without errors

## Recommended PR description
The PR description should make it easy to audit the submission. At minimum, include:
- Model identifier
- Model source (for example, Hugging Face model page or local checkpoint lineage)
- Benchmarks evaluated
- Whether the submission is retrieval or reranking
- Any non-default evaluation settings used in the run
- A short note on how the run was produced

Example:

```md
## Summary
- Model: `qwen3-embedding-4b-jua-jurisprudencia`
- Type: retrieval
- Benchmarks: `jua`, `juris-tcu`, `normas-tcu`

## Provenance
- Source: https://huggingface.co/ufca-llms/Qwen3-Embedding-4B-jua-jurisprudencia
- Evaluation command: `python -m jua.cli run --model qwen3-embedding-4b-jua-jurisprudencia --all_datasets`

## Notes
- This PR adds leaderboard metrics and model metadata only.
```

## Review expectations
A leaderboard PR should be reviewable from the repository contents alone. Reviewers should be able to answer:
- What model was evaluated?
- On which benchmarks?
- Is the model metadata consistent with the leaderboard entry?
- Are the files placed in the correct directories?
- Does the submission avoid committing raw artifacts that should remain outside the repository?

## Gradio app
The app reads `leaderboard/` by default. To run:
```
python3 leaderboard_app.py
```
