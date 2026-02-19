from argparse import ArgumentParser
import pandas as pd
import os

def convert_ulysses(corpus_path: str, queries_path: str, output_dir: str = "./data/ulysses"):
    """Converte arquivos de https://drive.google.com/file/d/1AstJwYZhiPUx-wKahwOxrTuJ7td5rLMi/view para formato de dataset"""
    # corpus_df = pd.read_csv(corpus_path)
    # corpus_df['_id'] = corpus_df['name']
    # corpus_df['title'] = corpus_df['name']

    # # Cria o diretório se não existir
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Converte para JSON
    # corpus_df[["_id", "title", "text"]].to_json(
    #     os.path.join(output_dir, "corpus.jsonl"),
    #     orient="records",
    #     lines=True,
    #     force_ascii=False
    # )

    queries_df = pd.read_csv(queries_path)
    queries_df['_id'] = queries_df['id']
    # cast _id to string
    queries_df['_id'] = queries_df['_id'].astype(str)
    queries_df['text'] = queries_df['query']

    # Converte para JSON
    queries_df[["_id", "text"]].to_json(
        os.path.join(output_dir, "queries.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False
    )

    # create qrels
    qrels = []
    for _, row in queries_df.iterrows():
        feedback = eval(row['user_feedback'])
        for item in feedback:
            relevance = {"i":0,"pr":1,"r":2}[item['class']]
            qrels.append({
                "query-id": str(row['_id']),
                "corpus-id": item['id'],
                "score": relevance
            })

    os.makedirs(os.path.join(output_dir, "qrels"), exist_ok=True)
    # Save qrels to file
    qrels_df = pd.DataFrame(qrels)
    qrels_df.to_csv(os.path.join(output_dir, "qrels", "test.tsv"), index=False,sep="\t")

if __name__ == "__main__":
    """python3 -m jua.create.convert_ulysses --corpus_path=data/bills_dataset.csv --queries_path=data/relevance_feedback_dataset.csv"""
    parser = ArgumentParser()

    parser.add_argument("--corpus_path", type=str, help="Caminho para o arquivo corpus do Ulysses", default=None)
    parser.add_argument("--queries_path", type=str, help="Caminho para o arquivo queries do Ulysses", default=None)
    parser.add_argument("--output_dir", type=str, default="./data/ulysses", help="Caminho de saída do dataset")

    args = parser.parse_args()

    convert_ulysses(args.corpus_path, args.queries_path, args.output_dir)