import json
import os

# 1. Pega a pasta onde este script está (jua/test)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Sobe uma pasta (para 'jua') e entra na pasta 'results'
RESULTS_FILE = os.path.join(SCRIPT_DIR, "..", "results", "anserini_bm25_hard.json")

def main():
    # Converte para caminho absoluto limpo para exibição
    caminho_absoluto = os.path.abspath(RESULTS_FILE)
    print(f"Buscando arquivo em: {caminho_absoluto}\n")

    if not os.path.exists(RESULTS_FILE):
        print(f"Erro: O arquivo nao foi encontrado no caminho acima!")
        print("Verifique se o nome do arquivo .json esta correto na variavel RESULTS_FILE.")
        return

    print("1/2. Lendo o arquivo de resultados...")
    
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_queries = len(data)
    total_docs_retornados = 0
    docs_por_query = []

    for qid, docs in data.items():
        count_docs = len(docs)
        total_docs_retornados += count_docs
        docs_por_query.append(count_docs)

    media_docs = (total_docs_retornados / total_queries) if total_queries > 0 else 0
    min_docs = min(docs_por_query) if docs_por_query else 0
    max_docs = max(docs_por_query) if docs_por_query else 0

    print("\n" + "="*50)
    print("CONTAGEM DETALHADA DOS RESULTADOS")
    print("="*50)
    print(f"- Total de Queries no arquivo: {total_queries}")
    print(f"- Total acumulado de Documentos: {total_docs_retornados}")
    print(f"- Media de documentos por query: {media_docs:.2f}")
    print(f"- Menor quantidade em uma query: {min_docs}")
    print(f"- Maior quantidade em uma query: {max_docs}")
    print("="*50)

if __name__ == "__main__":
    main()