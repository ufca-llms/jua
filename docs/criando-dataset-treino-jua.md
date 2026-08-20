# Criando o dataset de treino JUÁ do zero

Este guia descreve o fluxo usado para criar o dataset de treino a partir do CSV `data/jurisprudencia-selecionada.csv` do TCU.

O fluxo completo é:

1. gerar o dataset BEIR base (`corpus.jsonl`, `queries.jsonl`, `qrels`);
2. rodar BM25 com Anserini para recuperar candidatos negativos;
3. minerar exemplos positivos/negativos no formato JSONL de treino.

Observação: existe uma etapa opcional para gerar queries curtas sintéticas em `queries_with_questions_new.jsonl`. Ela foi usada para gerar `data/train_dataset_new.jsonl`, mas o arquivo `data/train_dataset.jsonl` foi gerado com as queries originais de `queries.jsonl`.

## 1. Preparar o CSV de entrada

O arquivo esperado é:

```bash
data/jurisprudencia-selecionada.csv
```

Ele deve estar separado por `|` e conter, pelo menos, as colunas usadas por `jua/dataset.py`:

```text
KEY
ENUNCIADO
EXCERTO
NUMACORDAO
ANOACORDAO
```

## 2. Gerar o dataset base

Rode:

```bash
python -m jua \
  --filepath data/jurisprudencia-selecionada.csv \
  --directory jua-dataset
```

Esse comando usa:

```text
jua/__main__.py
jua/dataset.py
```

O dataset base deve ter esta estrutura:

```text
jua-dataset/
  corpus.jsonl
  queries.jsonl
  qrels/
    train.tsv
    test.tsv
```

Os ids seguem este padrão:

```text
query-id:  JURISPRUDENCIA-SELECIONADA-47557-q
corpus-id: JURISPRUDENCIA-SELECIONADA-47557
```

Ou seja, a query termina com `-q`, e o documento positivo tem o mesmo id sem esse sufixo.

Verifique:

```bash
head -n 3 jua-dataset/queries.jsonl
head -n 3 jua-dataset/corpus.jsonl
head -n 5 jua-dataset/qrels/train.tsv
head -n 5 jua-dataset/qrels/test.tsv
```

Observação: se o script local gerar arquivos como `train_.tsv` e `test_.tsv`, renomeie ou regenere para o padrão esperado pelos scripts seguintes:

```bash
mv jua-dataset/qrels/train_.tsv jua-dataset/qrels/train.tsv
mv jua-dataset/qrels/test_.tsv jua-dataset/qrels/test.tsv
```

## 3. Subir o servidor BM25 Anserini

Em outro terminal, rode:

```bash
docker run \
  -p 8000:8000 \
  -e JAVA_TOOL_OPTIONS="-Xms1024m -Xmx8g" \
  --memory=12g \
  --memory-swap=12g \
  -it \
  beir/pyserini-fastapi
```

Confirme que o servidor está respondendo:

```bash
curl http://127.0.0.1:8000
```

## 4. Rodar BM25 para gerar candidatos negativos

O script principal é:

```text
jua/evaluate/bm25.py
```

O entrypoint CLI é:

```text
jua/evaluate/__main__.py
```

Rode:

```bash
python -m jua.evaluate bm25 \
  --dataset_path ./jua-dataset \
  --results_file results/anserini_bm25_hard.json \
  --server_url http://127.0.0.1:8000 \
  --chunk_size 100
```

Esse passo:

1. cria `jua-dataset/pyserini.jsonl`;
2. envia o corpus para o servidor `pyserini-fastapi`;
3. cria o índice Anserini;
4. busca os top-k documentos para cada query;
5. salva o ranking em `results/anserini_bm25_hard.json`;
6. calcula métricas em `results/anserini_bm25_hard_metrics.json`.

Verifique:

```bash
ls -lh results/anserini_bm25_hard.json
ls -lh results/anserini_bm25_hard_metrics.json
```

## 5. Gerar o JSONL final de treino

Agora use os resultados BM25 para minerar negativos:

```bash
python -m jua.train \
  --results_path results/anserini_bm25_hard.json \
  --dataset_path ./jua-dataset \
  --query_file queries.jsonl \
  --output_path ./data/train_dataset.jsonl \
  --alpha 0.01
```

Esse comando usa:

```text
jua/train/__main__.py
jua/train/train_dataset.py
```

Ele lê:

```text
results/anserini_bm25_hard.json
jua-dataset/corpus.jsonl
jua-dataset/queries.jsonl
jua-dataset/qrels/train.tsv
```

e escreve:

```text
data/train_dataset.jsonl
```

Cada linha segue o formato esperado pelo treino de embeddings:

```json
{
  "messages": [{"role": "user", "content": "query"}],
  "positive_messages": [[{"role": "user", "content": "documento positivo"}]],
  "negative_messages": [[{"role": "user", "content": "documento negativo"}]]
}
```

Verifique:

```bash
wc -l data/train_dataset.jsonl
head -n 1 data/train_dataset.jsonl
```

## 6. Etapa opcional: queries curtas sintéticas

Se quiser gerar uma variante com queries curtas do tipo palavras-chave, use:

```bash
touch jua-dataset/queries_with_questions_new.jsonl
python -m jua.create.generate_train_questions
```

Depois gere outro JSONL de treino apontando para esse arquivo:

```bash
python -m jua.train \
  --results_path results/anserini_bm25_hard.json \
  --dataset_path ./jua-dataset \
  --query_file queries_with_questions_new.jsonl \
  --output_path ./data/train_dataset_new.jsonl \
  --alpha 0.01
```

Essa variante corresponde ao padrão observado em `data/train_dataset_new.jsonl` e `data/combined_train_dataset.jsonl`.

## 7. Usar no treino do modelo

O arquivo final pode ser passado para o `swift sft`, por exemplo:

```bash
swift sft \
  --model Qwen/Qwen3-Embedding-4B \
  --task_type embedding \
  --train_type lora \
  --dataset ./data/train_dataset.jsonl \
  --loss_type infonce
```

O arquivo `data/run2.sh` contém comandos reais usados em experimentos anteriores, incluindo variações com múltiplos datasets.

## Checklist final

Ao fim do processo, estes arquivos devem existir:

```text
jua-dataset/corpus.jsonl
jua-dataset/queries.jsonl
jua-dataset/qrels/train.tsv
jua-dataset/qrels/test.tsv
jua-dataset/pyserini.jsonl
results/anserini_bm25_hard.json
results/anserini_bm25_hard_metrics.json
data/train_dataset.jsonl
```

## Scripts envolvidos

```text
jua/__main__.py
jua/dataset.py
jua/create/generate_train_questions.py
jua/evaluate/bm25.py
jua/evaluate/__main__.py
jua/train/__main__.py
jua/train/train_dataset.py
```
