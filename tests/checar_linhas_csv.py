import csv

NOME_DO_SEU_CSV = "jurisprudencia-selecionada.csv"  # Coloque o nome real do seu CSV
COLUNA_QUERY = "query"  # Nome da coluna que contém a pergunta (ou ajuste o índice)

total_linhas = 0
queries_unicas = set()
linhas_vazias_ou_invalidas = 0

with open(NOME_DO_SEU_CSV, 'r', encoding='utf-8', errors='replace') as f:
    # Testa se usa vírgula ou ponto-e-vírgula como separador
    sample = f.read(2048)
    f.seek(0)
    delimiter = ';' if sample.count(';') > sample.count(',') else ','
    
    reader = csv.DictReader(f, delimiter=delimiter)
    
    for idx, row in enumerate(reader, start=1):
        total_linhas += 1
        texto_query = row.get(COLUNA_QUERY, "").strip() if row.get(COLUNA_QUERY) else ""
        
        if not texto_query:
            linhas_vazias_ou_invalidas += 1
        else:
            queries_unicas.add(texto_query)

print("="*50)
print(f" DIAGNÓSTICO DO CSV FONTE:")
print(f"- Total de linhas no CSV: {total_linhas}")
print(f"- Queries ÚNICAS (sem duplicatas): {len(queries_unicas)}")
print(f"- Linhas com query vazia/inválida: {linhas_vazias_ou_invalidas}")
print(f"- Duplicatas descartáveis: {total_linhas - len(queries_unicas)}")
print("="*50)