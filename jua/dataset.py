from pathlib import Path
import pandas as pd
import re
from jua.bm25_scoring import BM25Scoring
import numpy as np
from tqdm.auto import tqdm
import json
import os

class Dataset:
    def __init__(self, filepath, sample_size=None):
        self.__file_path = Path(filepath)
        self.__df = self.load_data()

        if sample_size:
            self.__df = self.__df.sample(n=sample_size)
    
    def remove_tags(self, text):
        return re.sub(r'<[^>]*>', '', text)
    
    def load_data(self):
        self.__df = pd.read_csv(self.__file_path,sep="|",encoding="utf-8")
        self.__df["ENUNCIADO"] = self.__df["ENUNCIADO"].apply(self.remove_tags)
        self.__df["EXCERTO"] = self.__df["EXCERTO"].apply(self.remove_tags)
        # remove "SÚMULA TCU X[NUMSUMULA]:" from ENUNCIADO
        self.__df["ENUNCIADO"] = self.__df["ENUNCIADO"].apply(lambda x: re.sub(r"SÚMULA TCU (\d+):", "", x).strip())
        # remove examples with EXCERTO or ENUNCIADO empty
        self.__df = self.__df[self.__df["EXCERTO"].notna() & self.__df["ENUNCIADO"].notna()]
        # remove examples with NUMACORDAO or ANOACORDAO empty
        self.__df = self.__df[self.__df["NUMACORDAO"].notna() & self.__df["ANOACORDAO"].notna()]
        # remove examples with excerto in ["Não foi possível obter o conteúdo.","Digite o conteúdo do excerto."]
        self.__df = self.__df[~self.__df["EXCERTO"].isin(["Não foi possível obter o conteúdo.","Digite aqui o conteúdo do Excerto."])]
        self.__df["title"] = self.__df.apply(lambda x: f"{x['NUMACORDAO']}/{x['ANOACORDAO']}", axis=1)
        return self.__df
    
    def bm25_scoring(self):
        self.__scorer = BM25Scoring(self.__df["ENUNCIADO"].tolist())
        tqdm.pandas(desc="BM25 Scoring")
        self.__df["BM25_RANK"]= self.__df.progress_apply(lambda x: self.__scorer.score(x["ENUNCIADO"], self.__df.index.get_loc(x.name),len(self.__df)), axis=1)
        return self.__df
    
    def split_dataset(self):
        """
        Split the dataset into training and testing sets according to the BM25_RANK column.
        The testing will contain the top 10% of the dataset by BM25_RANK (harder questions)
        """
        self.__df = self.bm25_scoring()
        self.__df.to_csv("data/jurisprudencia-selecionada-bm25.csv", index=False)
        self.__df_test = self.__df.sort_values(by="BM25_RANK", ascending=False).head(int(len(self.__df) * 0.1))
        self.__df_train = self.__df.sort_values(by="BM25_RANK", ascending=False).iloc[int(len(self.__df) * 0.1):]
        return self.__df_train, self.__df_test
    
    def save_dataset(self, directory="jua-dataset"):
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory,'qrels'), exist_ok=True)
        self.split_dataset()
        # queries are the _id (numeric part of KEY column) and the text (ENUNCIADO column)
        queries = self.__df[["KEY", "ENUNCIADO"]].to_dict(orient="records")
        # rename KEY column to _id and ENUNCIADO column to text
        queries = [{"_id": query["KEY"], "text": query["ENUNCIADO"]} for query in queries]
        # save queries to jsonlines file
        with open(os.path.join(directory, "queries.jsonl"), "w", encoding="utf-8") as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")

        corpus = self.__df[["title", "KEY", "EXCERTO"]].to_dict(orient="records")
        # rename KEY column to _id and EXCERTO column to text
        corpus = [{"_id": corpus["KEY"], "title": corpus["title"], "text": corpus["EXCERTO"]} for corpus in corpus]
        # save corpus to jsonlines file
        with open(os.path.join(directory, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for doc in corpus:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        train_qrels_df = pd.DataFrame({
            "query_id": self.__df_train["KEY"],
            "corpus_id": self.__df_train["KEY"],
            "relevance": 1
        })
        # save train qrels to json file
        train_qrels_df.to_csv(os.path.join(directory,'qrels', "train.tsv"), index=False, sep="\t")
        test_qrels_df = pd.DataFrame({
            "query-id": self.__df_test["KEY"],
            "corpus-id": self.__df_test["KEY"],
            "score": 1
        })
        # save test qrels to json file
        test_qrels_df.to_csv(os.path.join(directory,'qrels', "test.tsv"), index=False, sep="\t")
    
    

    
        