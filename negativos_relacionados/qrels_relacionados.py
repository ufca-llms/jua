import os
import json
import csv

# 1. Como o script e o .jsonl estão na MESMA pasta:
MAP_PATH = "tcu_related_map.jsonl"

# 2. Subindo as pastas para achar o jua-dataset
# (A partir de 'negativos relacionados', voltamos uma pasta para chegar em 'jua')
TRAIN_TSV_PATH = os.path.join("..", "jua-dataset", "qrels", "train.tsv")
OUTPUT_QRELS = "qrels_gov_train.tsv"

def main():
    if not os.path.exists(MAP_PATH):
        print(f"❌ ERRO: O arquivo '{MAP_PATH}' não foi encontrado na pasta atual!")
        print(f"Pasta onde o terminal está rodando: {os.getcwd()}")
        return

    if not os.path.exists(TRAIN_TSV_PATH):
        print(f"❌ ERRO: Não encontrei o train.tsv em: {os.path.abspath(TRAIN_TSV_PATH)}")
        return

    print("1/3. Lendo o mapa do Gov...")
    gov_map = {}
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                key_id = str(item['key']).replace('-q', '')
                chaves_clean = [str(x).replace('-q', '') for x in item.get('chaves', [])]
                gov_map[key_id] = chaves_clean

    print("2/3. Processando QRELS e atribuindo scores (2 para direto, 0 para relacionado)...")
    qrels_linhas = []
    total_positivos = 0
    total_negativos = 0

    with open(TRAIN_TSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # Pula o cabeçalho se existir

        for row in reader:
            if not row or len(row) < 2:
                continue

            qid = str(row[0]).replace('-q', '')
            pos_doc_id = str(row[1]).replace('-q', '')

            # Positivo Direto -> Score 2
            qrels_linhas.append((f"{qid}-q", pos_doc_id, 2))
            total_positivos += 1

            # Negativos do Gov -> Score 0
            relacionados_ids = gov_map.get(pos_doc_id, [])
            for neg_id in relacionados_ids:
                if neg_id != pos_doc_id:
                    qrels_linhas.append((f"{qid}-q", neg_id, 0))
                    total_negativos += 1

    print("3/3. Salvando o arquivo final...")
    with open(OUTPUT_QRELS, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid, doc_id, score in qrels_linhas:
            writer.writerow([qid, doc_id, score])

    tamanho_mb = os.path.getsize(OUTPUT_QRELS) / (1024 * 1024)
    print("\n" + "="*50)
    print(f"SUCESSO! Arquivo gerado: {OUTPUT_QRELS} ({tamanho_mb:.2f} MB)")
    print(f"- Positivos (Score 2): {total_positivos}")
    print(f"- Negativos do Gov (Score 0): {total_negativos}")
    print("="*50)

if __name__ == "__main__":
    main()