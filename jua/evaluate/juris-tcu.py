from argparse import ArgumentParser
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
import json
import pandas as pd
import os

def evaluate_juris_tcu(results_path: str, dataset_path: str):
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    query_path = os.path.join(dataset_path, "queries.jsonl")
    qrels_path = os.path.join(dataset_path,'qrels', "test.tsv")
    print(f"Loading dataset from {corpus_path}, {query_path}, {qrels_path}")
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path, 
        query_file=query_path, 
        qrels_file=qrels_path).load_custom()
    
    results = json.load(open(results_path, "r"))

    results_df = []

    print("All queries results:")
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10, 100],ignore_identical_ids=False)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 3, 5, 10, 100], metric="mrr")
    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}, MRR: {mrr}")

    results_df.append({
        "group": "all",
        "ndcg@10": ndcg["NDCG@10"],
        "map@10": _map["MAP@10"],
        "mrr@10": mrr["MRR@10"]
     })

    for i in range(0, 150, 50):
        results_subset = {k: v for k, v in list(results.items())[i:i+50]}

        print(f"Evaluating queries {i} to {i+50}...")
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_subset, [1, 3, 5, 10, 100],ignore_identical_ids=False)
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results_subset, [1, 3, 5, 10, 100], metric="mrr")
        print(f"Queries {i}-{i+50}: NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}, MRR: {mrr}")

        results_df.append({
            "group": f"{i} to {i+50}",
            "ndcg@10": ndcg["NDCG@10"],
            "map@10": _map["MAP@10"],
            "mrr@10": mrr["MRR@10"]
        })

    df = pd.DataFrame(results_df)
    print(df.head())
    df.to_csv(f"results/juris-tcu_evaluation.csv", index=False)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()
    
    results_path = args.results_path
    dataset_path = args.dataset_path
    
    print(f"Evaluating results in {results_path} with dataset in {dataset_path}...")

    evaluate_juris_tcu(results_path, dataset_path)