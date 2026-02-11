import pandas as pd
import re

def remove_tags(text):
    return re.sub(r'<[^>]*>', '', text)

corpus = pd.read_csv("./data/juris-tcu/doc.csv")
queries = pd.read_csv("./data/juris-tcu/query.csv")
qrels = pd.read_csv("data/juris-tcu/qrel.csv")

# keep only numbers from KEY column
corpus["_id"] = corpus["KEY"].str.extract("(\d+)").astype(str)
corpus["title"] = corpus.apply(lambda x: f"{x['NUMACORDAO']}/{x['ANOACORDAO']}", axis=1)
corpus["text"] = corpus["EXCERTO"]
corpus["text"] = corpus["text"].apply(remove_tags)
print(corpus[["KEY", "_id", "title", "text"]].head())

corpus["_id"] = corpus["_id"].astype(str)
corpus = corpus[["_id", "title", "text"]]
corpus.to_json("./data/juris-tcu/beir/corpus.jsonl", orient="records", lines=True, force_ascii=False)

# {"_id":"JURISPRUDENCIA-SELECIONADA-189673-q","text":"Não se exige a observância do contraditório e da ampla defesa na apreciação da legalidade de ato de concessão inicial de aposentadoria, reforma e pensão, bem como de ato que lhes altere os fundamentos legais, salvo se decorrido prazo igual ou superior a cinco anos, a partir do ingresso do ato no TCU, hipótese em que ocorre o registro tácito, tornando-se obrigatórias, em caso de revisão de ofício, as garantias do contraditório e da ampla defesa, quando nele verificada irregularidade e desde que tenha ingressado há menos de dez anos no TCU, ou, ainda, no caso de imputação de má-fé ao interessado, independentemente do prazo decorrido."}
# queries rename ID to _id and TEXT to text
queries["_id"] = queries["ID"]
# _id to str
queries["_id"] = queries["_id"].astype(str)

queries["text"] = queries["TEXT"]
queries = queries[["_id", "text"]]
queries.to_json("./data/juris-tcu/beir/queries.jsonl", orient="records", lines=True, force_ascii=False)

# qrels to query-id	corpus-id	score
qrels["query-id"] = qrels["QUERY_ID"]
qrels["corpus-id"] = qrels["DOC_ID"]
qrels["query-id"] = qrels["query-id"].astype(str)
qrels["corpus-id"] = qrels["corpus-id"].astype(str)
qrels["score"] = qrels["SCORE"]

qrels = qrels[["query-id", "corpus-id", "score"]]
qrels.to_csv("./data/juris-tcu/beir/qrels/test.tsv", sep="\t", index=False)