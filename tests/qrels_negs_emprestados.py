import os
import json
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
JUA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "jua-dataset"))

# Entradas
ALL_TSV_PATH = os.path.join(JUA_DIR, "qrels", "all.tsv")
LOG_EMPRESTADOS_PATH = os.path.join(DATA_DIR, "linhas_emprestadas.json")

# Saída
OUTPUT_EMPRESTADOS_PATH = os.path.join(DATA_DIR, "qrels_emprestados_oficial.tsv")

def main():
    print("1/2. Carregando mapeamento de linhas emprestadas...")

    if not os.path.exists(LOG_EMPRESTADOS_PATH):
        print(f" Arquivo de log não encontrado em: {LOG_EMPRESTADOS_PATH}")
        return

    with open(LOG_EMPRESTADOS_PATH, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
        # Cria um mapa: linha_id -> query_snippet (para conferencia)
        linhas_emprestadas_map = {item["linha_id"]: item.get("query_snippet", "") for item in log_data}

    print(f"   {len(linhas_emprestadas_map)} queries identificadas como emprestadas.")

    print("\n2/2. Cruzando com o 'all.tsv' para resgatar os Query IDs oficiais...")

    linhas_qrels = []
    total_emprestados_acumulado = 0

    with open(ALL_TSV_PATH, 'r', encoding='utf-8') as f_tsv:
        reader = csv.reader(f_tsv, delimiter='\t')
        header = next(reader, None)  # Pula o cabecalho do all.tsv

        for num_linha, row in enumerate(reader, start=1):
            if not row or len(row) < 2:
                continue

            # Se a linha atual esta na lista de emprestados
            if num_linha in linhas_emprestadas_map:
                query_id = row[0]
                pos_doc_id = row[1]
                qtd_emprestada = 3  # Padrao fixado no script de fusao

                linhas_qrels.append([
                    query_id,
                    pos_doc_id,
                    qtd_emprestada,
                    "alpha05",
                    num_linha
                ])
                total_emprestados_acumulado += qtd_emprestada

    # Grava o relatório oficial
    with open(OUTPUT_EMPRESTADOS_PATH, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["query_id", "pos_doc_id", "qtd_negativos_emprestados", "fonte", "linha_original"])
        writer.writerows(linhas_qrels)

    print("\n" + "="*50)
    print(" QRELS DE EMPRESTADOS GERADO COM SUCESSO!")
    print("="*50)
    print(f"- Total de queries resgatadas: {len(linhas_qrels)}")
    print(f"- Negativos emprestados no total: {total_emprestados_acumulado}")
    print(f"- Arquivo oficial salvo em: {OUTPUT_EMPRESTADOS_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()