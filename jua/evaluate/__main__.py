from argparse import ArgumentParser
import os
from beir.datasets.data_loader import GenericDataLoader
from jua.evaluate.bm25 import evaluate_bm25
from jua.evaluate.dense_hf import evaluate_dense_hf
from jua.evaluate.sbert import evaluate_sbert
from jua.evaluate.openai_embeddings import evaluate_openai_embeddings
from jua.evaluate.reranking_dense import evaluate_reranking_dense
from jua.evaluate.reranking_monot5 import evaluate_reranking_monot5

def main(model_name: str, model_type: str, dataset_path: str):
    corpus, queries, qrels = load_dataset(dataset_path)

    
    if model_type == "bm25":
        evaluate_bm25(corpus, queries, qrels)
    elif model_type == "dense_hf":
        evaluate_dense_hf(model_name, corpus, queries, qrels)
    elif model_type == "sbert":
        evaluate_sbert(model_name, corpus, queries, qrels)
    elif model_type == "openai":
        evaluate_openai_embeddings(model_name, corpus, queries, qrels)
    elif model_type == "reranking_dense":
        evaluate_reranking_dense(corpus, queries, qrels,model_name)
    elif model_type == "reranking_monot5":
        evaluate_reranking_monot5(corpus, queries, qrels, model_name, token_false="▁no", token_true="▁yes")

def load_dataset(dataset_path: str):
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    query_path = os.path.join(dataset_path, "queries.jsonl")
    qrels_path = os.path.join(dataset_path,'qrels', "test.tsv")
    print(f"Loading dataset from {corpus_path}, {query_path}, {qrels_path}")
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path, 
        query_file=query_path, 
        qrels_file=qrels_path).load_custom()

    return corpus, queries, qrels


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name",default=None)
    parser.add_argument("--model_type", type=str, help="Model type", default="bm25")
    parser.add_argument("--dataset_path", type=str, default="./jua-dataset",help="Dataset path")

    args = parser.parse_args()

    main(args.model_name, args.model_type, args.dataset_path)