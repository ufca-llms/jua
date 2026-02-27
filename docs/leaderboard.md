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

## Gradio app
The app reads `leaderboard/` by default. To run:
```
python3 leaderboard_app.py
```
