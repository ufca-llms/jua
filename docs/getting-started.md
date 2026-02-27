# Getting Started

## Quick install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a model on one benchmark
```
python -m jua.cli run --model qwen3-embedding-0.6b --benchmark jua
```

## Run on all datasets
```
python -m jua.cli run --model qwen3-embedding-0.6b --all_datasets
```

## Leaderboard
Metrics are saved in `leaderboard/<model>/` and raw results in `results/leaderboard/<model>/`.
