import os
import json
import csv

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "jua-dataset")
MAP_FILE = os.path.join(BASE_DIR, "tcu_related_map.jsonl")  # Aceita .json ou sem extensão
OUTPUT_FILE = os.path.join(BASE_DIR, "train_dataset_jua_com_relacionados.jsonl")

def load_gov_map(map_path):
    """Carrega o arquivo do MAP do Gov/TCU"""
    gov_map_dict = {}
    
    if not os.path.exists(map_path) and os.path.exists(map_path + ".json"):
        map_path = map_path + ".json"
        
    print(f"Carregando o mapa do GOV de: {map_path}")
    
    with open(map_path, 'r', encoding='utf-8') as f:
        if map_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gov_map_dict[str(item["key"]).replace('-q', '')] = item.get("chaves", [])
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    gov_map_dict[str(item["key"]).replace('-q', '')] = item.get("chaves", [])
            elif isinstance(data, dict):
                gov_map_dict = {str(k).replace('-q', ''): [str(x).replace('-q', '') for x in v.get("chaves", [])] if isinstance(v, dict) else v for k, v in data.items()}

    return gov_map_dict

def main():
    # 1. Carregar Corpus (doc_id -> texto)
    print("1/4. Carregando corpus.jsonl...")
    corpus = {}
    corpus_file = os.path.join(DATASET_PATH, "corpus.jsonl")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                clean_id = str(data['_id']).replace('-q', '')
                corpus[clean_id] = data['text']

    # 2. Carregar Queries (query_id -> texto)
    print("2/4. Carregando queries.jsonl...")
    queries = {}
    queries_file = os.path.join(DATASET_PATH, "queries.jsonl")
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                clean_qid = str(data['_id']).replace('-q', '')
                queries[clean_qid] = data['text']

    # 3. Carregar o Mapa do GOV/TCU
    print("3/4. Mapeando relações do TCU...")
    gov_map = load_gov_map(MAP_FILE)

    # 4. Cruzar dados e montar a estrutura com o padrão do professor
    print("4/4. Gerando dataset de treino mantendo todas as queries...")
    qrels_file = os.path.join(DATASET_PATH, "qrels", "train.tsv")
    dataset_treino = []
    
    queries_com_negativos = 0
    queries_sem_negativos = 0
    total_negativos_encontrados = 0

    with open(qrels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # Pula cabeçalho se houver

        for row in reader:
            if not row or len(row) < 2:
                continue

            qid = str(row[0]).replace('-q', '')
            pos_doc_id = str(row[1]).replace('-q', '')

            # Valida existência no corpus e nas queries
            if qid in queries and pos_doc_id in corpus:
                query_text = queries[qid]
                pos_text = corpus[pos_doc_id]

                # Busca as chaves relacionadas no MAP usando o ID do documento positivo
                relacionados_ids = gov_map.get(pos_doc_id, [])

                # Extrai apenas os negativos válidos existentes no corpus
                negativos_textos = []
                for neg_id in relacionados_ids:
                    neg_clean_id = str(neg_id).replace('-q', '')
                    if neg_clean_id != pos_doc_id and neg_clean_id in corpus:
                        negativos_textos.append(corpus[neg_clean_id])

                if negativos_textos:
                    queries_com_negativos += 1
                    total_negativos_encontrados += len(negativos_textos)
                else:
                    queries_sem_negativos += 1

                # MANTÉM A QUERY NO DATASET, com ou sem negativos
                dataset_treino.append({
                    "messages": query_text,
                    "positive_messages": pos_text,
                    "negative_messages": negativos_textos
                })

    # 5. Salvar resultado final em JSONL
    print(f"\nSalvando resultado em: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset_treino:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Exibe relatório detalhado do dataset
    total_queries = len(dataset_treino)
    media_negativos_geral = (total_negativos_encontrados / total_queries) if total_queries else 0
    media_negativos_validos = (total_negativos_encontrados / queries_com_negativos) if queries_com_negativos else 0

    print("\n" + "="*60)
    print("SUCESSO! RELATÓRIO DO DATASET GERADO (PADRÃO PROFESSOR):")
    print(f"- Total de queries registradas: {total_queries}")
    print(f"- Queries COM negativos do Gov: {queries_com_negativos}")
    print(f"- Queries SEM negativos do Gov (mantidas): {queries_sem_negativos}")
    print(f"- Total absoluto de negativos vinculados: {total_negativos_encontrados}")
    print(f"- Média de negativos (considerando todas as queries): {media_negativos_geral:.2f}")
    print(f"- Média de negativos (apenas entre queries que possuem negativos): {media_negativos_validos:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()