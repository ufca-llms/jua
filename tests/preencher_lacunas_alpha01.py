import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

ALPHA_01_PATH = os.path.join(DATA_DIR, "train_dataset_new_01.jsonl")
ALPHA_05_PATH = os.path.join(DATA_DIR, "train_dataset_new_05.jsonl")

# Arquivos de saída
HYBRID_OUTPUT_PATH = os.path.join(DATA_DIR, "train_dataset_hybrid.jsonl")
MODIFIED_IDS_PATH = os.path.join(DATA_DIR, "linhas_emprestadas.json")

MAX_EMPRESTADOS = 3

def main():
    print("1/2. Processando arquivos linha por linha simultaneamente...\n")

    if not os.path.exists(ALPHA_01_PATH) or not os.path.exists(ALPHA_05_PATH):
        print("Arquivos de entrada não encontrados na pasta 'data/'!")
        return

    linhas_modificadas = []  # Guarda os IDs (número da linha) onde houve empréstimo
    total_linhas = 0
    recuperadas = 0

    with open(ALPHA_01_PATH, 'r', encoding='utf-8') as f_01, \
         open(ALPHA_05_PATH, 'r', encoding='utf-8') as f_05, \
         open(HYBRID_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        for num_linha, (line_01, line_05) in enumerate(zip(f_01, f_05), start=1):
            if not line_01.strip():
                continue

            total_linhas += 1
            item_01 = json.loads(line_01)
            negs_01 = item_01.get("negative_messages", [])

            # Se o Alpha 0.01 NÃO tem negativos (lista vazia)
            if len(negs_01) == 0:
                item_05 = json.loads(line_05)
                negs_05 = item_05.get("negative_messages", [])

                if len(negs_05) > 0:
                    # Empresta 3 negativos do Alpha 0.05
                    item_01["negative_messages"] = negs_05[:MAX_EMPRESTADOS]
                    recuperadas += 1
                    
                    # Salva a linha (1-indexed) e o snippet da pergunta para validação futura
                    q_text = item_01.get("messages", [{}])[0].get("content", "")[:60]
                    linhas_modificadas.append({
                        "linha_id": num_linha,
                        "query_snippet": q_text
                    })

            # Escreve a linha (seja intacta ou modificada) mantendo a ordem original
            f_out.write(json.dumps(item_01, ensure_ascii=False) + "\n")

    # Salva o arquivo com as linhas que receberam empréstimo
    with open(MODIFIED_IDS_PATH, 'w', encoding='utf-8') as f_log:
        json.dump(linhas_modificadas, f_log, indent=2, ensure_ascii=False)

    print("="*50)
    print("RESULTADO DO PROCESSAMENTO LINHA POR LINHA")
    print("="*50)
    print(f"- Total de linhas mantidas intactas: {total_linhas}")
    print(f"- Linhas que receberam 3 negativos do Alpha 05: {recuperadas}")
    print(f"- Arquivo Híbrido salvo em: {HYBRID_OUTPUT_PATH}")
    print(f"- Mapeamento das linhas alteradas salvo em: {MODIFIED_IDS_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()