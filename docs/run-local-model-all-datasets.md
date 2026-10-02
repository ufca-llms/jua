# Como rodar um modelo local em todos os datasets

Este guia mostra como clonar o JUÁ, registrar um modelo local no registry e executar a avaliação em todos os datasets usando 1 GPU e `batch_size` ajustável.

## 1. Clonar o repositório

```bash
git clone git@github.com:ufca-llms/jua.git
cd jua
```

Se você preferir HTTPS:

```bash
git clone https://github.com/ufca-llms/jua.git
cd jua
```

## 2. Criar o ambiente

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install sentence-transformers torch
```

Se o `torch` instalado não detectar CUDA, instale a versão compatível com sua GPU/CUDA conforme a documentação do PyTorch.

## 3. Registrar o modelo local

Abra o arquivo:

```bash
jua/models/registry.json
```

Adicione uma nova entrada à lista JSON. Para um modelo local compatível com `sentence-transformers`, use `adapter: "sbert"`.

Exemplo:

```json
{
  "id": "meu-modelo-local",
  "adapter": "sbert",
  "model_name": "/caminho/absoluto/para/meu/modelo",
  "batch_size": 32,
  "devices": ["cuda:0"],
  "meta": {
    "name": "Meu Modelo Local",
    "provider": "local",
    "description": "Modelo local avaliado no benchmark JUÁ",
    "model_type": ["dense-retrieval"],
    "modalities": ["text"],
    "languages": ["pt"]
  }
}
```

Notas:

- `id` é o nome usado na CLI.
- `model_name` pode ser um caminho local, por exemplo `/home/user/models/meu-modelo`, ou um identificador Hugging Face.
- `batch_size` no registry é o valor padrão, mas pode ser sobrescrito na execução.
- `devices: ["cuda:0"]` define a GPU padrão do modelo, mas também é possível passar `--devices cuda:0` na CLI.

Depois, confirme que o modelo aparece:

```bash
python -m jua.cli list-models
```

## 4. Conferir os datasets registrados

```bash
python -m jua.cli list-datasets
```

Atualmente, `--all_datasets` executa todos os datasets registrados em `jua/benchmarks/registry.json`.

## 5. Rodar em todos os datasets com 1 GPU

Ajuste estas variáveis:

```bash
export MODEL_ID="meu-modelo-local"
export BATCH_SIZE=32
export GPU_ID=0
```

Execute:

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python -m jua.cli run \
  --model "$MODEL_ID" \
  --all_datasets \
  --devices cuda:0 \
  --batch_size "$BATCH_SIZE" \
  --output_dir leaderboard \
  --results_output_dir results/leaderboard
```

Mesmo usando `CUDA_VISIBLE_DEVICES=$GPU_ID`, mantenha `--devices cuda:0`. Dentro do processo, a GPU visível passa a ser indexada como `cuda:0`.

## 6. Ajustar batch size

Para aumentar ou diminuir o batch size, basta alterar `BATCH_SIZE`:

```bash
export BATCH_SIZE=16
CUDA_VISIBLE_DEVICES=0 python -m jua.cli run \
  --model "$MODEL_ID" \
  --all_datasets \
  --devices cuda:0 \
  --batch_size "$BATCH_SIZE"
```

Se ocorrer erro de memória na GPU, reduza o `BATCH_SIZE`, por exemplo `16`, `8`, `4` ou `2`.

## 7. Arquivos gerados

Os resultados principais ficam em:

```text
leaderboard/<modelo>/
```

com um arquivo JSON por dataset e um `model_meta.json`.

Os resultados detalhados ficam em:

```text
results/leaderboard/<modelo>/
```

## 8. Exemplo completo

```bash
git clone git@github.com:ufca-llms/jua.git
cd jua

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install sentence-transformers torch

# Edite jua/models/registry.json e adicione o modelo com id "meu-modelo-local".

python -m jua.cli list-models
python -m jua.cli list-datasets

export MODEL_ID="meu-modelo-local"
export BATCH_SIZE=32
export GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python -m jua.cli run \
  --model "$MODEL_ID" \
  --all_datasets \
  --devices cuda:0 \
  --batch_size "$BATCH_SIZE" \
  --output_dir leaderboard \
  --results_output_dir results/leaderboard
```
