# jua
JUÁ — An Information Retrieval Corpus of Public Audit Court Rulings

## Quick Start
### Dataset generation
```
python -m jua --filepath data/jurisprudencia-selecionada.csv --directory jua-dataset
```

### Evaluation (subcommands)
#### BM25 (Anserini via pyserini-fastapi)
1) Start the server:
```
docker run -p 8000:8000 -e JAVA_TOOL_OPTIONS="-Xms1024m -Xmx8g" --memory=12g --memory-swap=12g -it beir/pyserini-fastapi
```
2) Run evaluation:
```
python -m jua.evaluate bm25 --dataset_path ./data/ulysses --results_file results/anserini_bm25_ulysses.json
```

#### Dense HF
```
python -m jua.evaluate dense-hf --model_name sentence-transformers/all-MiniLM-L6-v2 --dataset_path ./jua-dataset
```

#### SBERT
```
python -m jua.evaluate sbert --model_name KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 --batch_size 128 --dataset_path ./jua-dataset
```

#### OpenAI/Gemini embeddings
```
python -m jua.evaluate openai --model_name text-embedding-3-small --batch_size 128 --dataset_path ./jua-dataset
```

#### Reranking (dense)
```
python -m jua.evaluate rerank-dense --model_name text-embedding-3-small --results_file results/anserini_bm25_hard.json --dataset_path ./jua-dataset
```

#### Reranking (MonoT5)
```
python -m jua.evaluate rerank-monot5 --model_name castorini/monot5-base-msmarco-10k --batch_size 128 --dataset_path ./jua-dataset
```

## Legacy CLI
The previous `--model_type` CLI is still supported for compatibility:
```
python -m jua.evaluate --model_type bm25 --dataset_path ./jua-dataset
```
