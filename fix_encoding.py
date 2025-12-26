import json

# Ler o arquivo e decodificar os caracteres Unicode escapados
with open('jua-dataset/queries_with_questions.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Reescrever com ensure_ascii=False
with open('jua-dataset/queries_with_questions.jsonl', 'w', encoding='utf-8') as f:
    for line in lines:
        if line.strip():  # Ignorar linhas vazias
            data = json.loads(line)  # Isso já decodifica os caracteres Unicode
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

