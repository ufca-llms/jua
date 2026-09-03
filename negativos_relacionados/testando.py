#TESTANDO PRA VER A ESTRUTURA DE CADA JSON DO JSONL

import json 

def linhas_do_json(arquivo, id_linha=1):
    with open(arquivo, "r", encoding="utf-8") as f:
        if id_linha == None or id_linha == 1:
            linha = f.readline()
        else:
            linha = None
            for idx, l in enumerate(f, 1):
                if idx == id_linha:
                    linha = l
                    break

        if not linha:
            print("O arquivo está vazio!")
            return None

        unidade = json.loads(linha)
        #linecache.clearcache()

        return unidade

def mostrar_json(unidade):
    if unidade is not None:

        original = unidade.get("messages")
        print(original)
        print("-" * 60)

        positivos = unidade.get("positive_messages")
        print(positivos)
        print("-" * 60)

        negativos = unidade.get("negative_messages")
        if not isinstance(negativos, list):
            negativos = [negativos]
        for id, mensagem in enumerate(negativos):
            print(str(id+1) + " -> " + str(mensagem))

        print("-" * 120)

def doc_pra_json(unidade):
    if unidade is None: return {}

    negativos = unidade.get("negative_messages", [])
    if not isinstance(negativos, list): negativos = [negativos]

    return {
        "messages": unidade.get("messages"),
        "positive_messages": unidade.get("positive_messages"),
        "negative_messages": negativos,
    }

exemplo = linhas_do_json("train_dataset_gov.jsonl",1)
mostrar_json(exemplo)