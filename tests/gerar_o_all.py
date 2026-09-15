import os
import csv

# Pega o diretório atual onde este script está (jua/test)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sobe um nível para a raiz 'jua/' e entra em 'jua-dataset/qrels/'
QRELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "jua-dataset", "qrels"))

TRAIN_PATH = os.path.join(QRELS_DIR, "train.tsv")
TEST_PATH = os.path.join(QRELS_DIR, "test.tsv")
ALL_PATH = os.path.join(QRELS_DIR, "all.tsv")

def main():
    print(f"Procurando arquivos em: {QRELS_DIR}\n")
    
    linhas_unicas = set()
    header = None

    for filepath in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(filepath):
            print(f"Arquivo nao encontrado: {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            curr_header = next(reader, None)
            if not header and curr_header:
                header = curr_header

            for row in reader:
                if row:
                    linhas_unicas.add(tuple(row))

    with open(ALL_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if header:
            writer.writerow(header)
        for row in linhas_unicas:
            writer.writerow(row)

    print("="*50)
    print(f"SUCESSO! 'all.tsv' gerado com {len(linhas_unicas)} pares!")
    print(f"Salvo em: {ALL_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()