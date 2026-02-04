"""
This example shows how to evaluate Anserini-BM25 in BEIR.
Since Anserini uses Java-11, we would advise you to use docker for running Pyserini.
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull beir/pyserini-fastapi
2. docker build -t pyserini-fastapi .
3. docker run -p 8000:8000 -it --rm pyserini-fastapi

Once the docker container is up and running in local, now run the code below.
This code doesn't require GPU to run.

Usage: python evaluate_anserini_bm25.py
"""

import json
import logging
import os
import pathlib
import random
from tqdm import tqdm
import requests

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


data_path = "./jua-dataset"
corpus_path = os.path.join(data_path, "corpus.jsonl")
query_path = os.path.join(data_path, "queries.jsonl")
qrels_path = os.path.join(data_path,'qrels', "test_hard.tsv")
print(f"Loading dataset from {corpus_path}, {query_path}, {qrels_path}")
corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()


#### Convert BEIR corpus to Pyserini Format #####
pyserini_jsonl = "pyserini.jsonl"
with open(os.path.join(data_path, pyserini_jsonl), "w", encoding="utf-8") as fOut:
    for doc_id in corpus:
        title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
        data = {"id": doc_id, "title": title, "contents": text}
        json.dump(data, fOut)
        fOut.write("\n")

#### Download Docker Image beir/pyserini-fastapi ####
#### Locally run the docker Image + FastAPI ####
docker_beir_pyserini = "http://127.0.0.1:8000"

#### Upload Multipart-encoded files ####
with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
    r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

#### Index documents to Pyserini #####
index_name = "beir/jua-dataset"  # beir/scifact

r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

#### Retrieve documents from Pyserini #####
retriever = EvaluateRetrieval()
# create chunks of 100 queries for batch processing
chunk_size = 100
# results = json.load(open("results/anserini_bm25_partial.json", "r"))
results = {}
for i in tqdm(range(len(results), len(queries), chunk_size)):
    
    chunk_queries = dict(list(queries.items())[i:i + chunk_size])

    qids = list(chunk_queries.keys())
    query_texts = [chunk_queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

    #### Retrieve pyserini results (format of results is identical to qrels)
    results.update(json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"])

    # json.dump(results, open("results/anserini_bm25_partial.json", "w"))

#### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
# results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

#### Check if query_id is in results i.e. remove it from docs incase if it appears ####
#### Quite Important for ArguAna and Quora ####
for query_id in results:
    if query_id in results[query_id]:
        results[query_id].pop(query_id, None)

json.dump(results, open("results/anserini_bm25_hard.json", "w"))

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

print(f"NDCG: {ndcg}")
print(f"_MAP: {_map}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"MRR: {mrr}")