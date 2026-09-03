from argparse import ArgumentParser
import os
from beir.datasets.data_loader import GenericDataLoader
from jua.evaluate.bm25 import evaluate_bm25
from jua.evaluate.dense_hf import evaluate_dense_hf
from jua.evaluate.sbert import evaluate_sbert
from jua.evaluate.openai_embeddings import evaluate_openai_embeddings
from jua.evaluate.reranking_dense import evaluate_reranking_dense
from jua.evaluate.reranking_monot5 import evaluate_reranking_monot5

DEFAULT_RESULTS_FILE = "results/anserini_bm25_hard.json"

def run_bm25(dataset_path: str, results_file: str, server_url: str, index_name: str | None, chunk_size: int):
    corpus, queries, qrels = load_dataset(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    evaluate_bm25(
        corpus,
        queries,
        qrels,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        results_file=results_file,
        server_url=server_url,
        index_name=index_name,
        chunk_size=chunk_size,
    )

def run_dense_hf(model_name: str, dataset_path: str):
    corpus, queries, qrels = load_dataset(dataset_path)
    evaluate_dense_hf(model_name, corpus, queries, qrels)

def run_sbert(model_name: str, dataset_path: str, batch_size: int):
    corpus, queries, qrels = load_dataset(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    evaluate_sbert(model_name, dataset_name, corpus, queries, qrels, batch_size)

def run_openai(model_name: str, dataset_path: str, batch_size: int):
    corpus, queries, qrels = load_dataset(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    evaluate_openai_embeddings(model_name, dataset_name, corpus, queries, qrels, batch_size)

def run_reranking_dense(model_name: str, dataset_path: str, results_file: str, embeddings_dir: str | None):
    _corpus, _queries, qrels = load_dataset(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    evaluate_reranking_dense(qrels, model_name, results_file, dataset_name, embeddings_dir=embeddings_dir)

def run_reranking_monot5(model_name: str, dataset_path: str, batch_size: int):
    corpus, queries, qrels = load_dataset(dataset_path)
    evaluate_reranking_monot5(
        corpus,
        queries,
        qrels,
        model_name,
        token_false="▁no",
        token_true="▁yes",
        batch_size=batch_size,
    )

def load_dataset(dataset_path: str):
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    query_path = os.path.join(dataset_path, "queries.jsonl")
    #qrels_path = os.path.join(dataset_path,'qrels', "test.tsv")
    qrels_path = os.path.join(dataset_path,'qrels', "train.tsv")
    print(f"Loading dataset from {corpus_path}, {query_path}, {qrels_path}")
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path, 
        query_file=query_path, 
        qrels_file=qrels_path).load_custom()

    return corpus, queries, qrels

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate retrieval models with subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Common args
    def add_dataset_path(p):
        p.add_argument("--dataset_path", type=str, default="./jua-dataset", help="Dataset path")

    # bm25
    bm25_parser = subparsers.add_parser("bm25", help="Evaluate BM25 baseline (Anserini)")
    bm25_parser.add_argument("--results_file", type=str, default="results/anserini_bm25.json", help="Results file")
    bm25_parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000", help="Pyserini FastAPI URL")
    bm25_parser.add_argument("--index_name", type=str, default=None, help="Index name (default: beir/<dataset>)")
    bm25_parser.add_argument("--chunk_size", type=int, default=100, help="Batch size for queries")
    add_dataset_path(bm25_parser)

    # dense_hf
    dense_hf_parser = subparsers.add_parser("dense-hf", help="Evaluate Dense HF model")
    dense_hf_parser.add_argument("--model_name", type=str, required=True, help="Model name")
    add_dataset_path(dense_hf_parser)

    # sbert
    sbert_parser = subparsers.add_parser("sbert", help="Evaluate SBERT model")
    sbert_parser.add_argument("--model_name", type=str, required=True, help="Model name")
    sbert_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    add_dataset_path(sbert_parser)

    # openai embeddings
    openai_parser = subparsers.add_parser("openai", help="Evaluate OpenAI/Gemini embeddings")
    openai_parser.add_argument("--model_name", type=str, required=True, help="Model name")
    openai_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    add_dataset_path(openai_parser)

    # reranking dense
    rerank_dense_parser = subparsers.add_parser("rerank-dense", help="Dense reranking over prior results")
    rerank_dense_parser.add_argument("--model_name", type=str, required=True, help="Model name")
    rerank_dense_parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE, help="Results file for reranking")
    rerank_dense_parser.add_argument("--embeddings_dir", type=str, default=None, help="Path to embeddings directory (overrides default)")
    add_dataset_path(rerank_dense_parser)

    # reranking monoT5
    rerank_monot5_parser = subparsers.add_parser("rerank-monot5", help="MonoT5 reranking over prior results")
    rerank_monot5_parser.add_argument("--model_name", type=str, required=True, help="Model name")
    rerank_monot5_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    add_dataset_path(rerank_monot5_parser)

    # Backward-compatible args (model_type)
    parser.add_argument("--model_name", type=str, help="Model name", default=None)
    parser.add_argument("--model_type", type=str, help="Model type", default=None)
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path (legacy)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (legacy)")
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE, help="Results file for reranking (legacy)")

    args = parser.parse_args()

    if args.command == "bm25":
        run_bm25(args.dataset_path, args.results_file, args.server_url, args.index_name, args.chunk_size)
    elif args.command == "dense-hf":
        run_dense_hf(args.model_name, args.dataset_path)
    elif args.command == "sbert":
        run_sbert(args.model_name, args.dataset_path, args.batch_size)
    elif args.command == "openai":
        run_openai(args.model_name, args.dataset_path, args.batch_size)
    elif args.command == "rerank-dense":
        run_reranking_dense(args.model_name, args.dataset_path, args.results_file, args.embeddings_dir)
    elif args.command == "rerank-monot5":
        run_reranking_monot5(args.model_name, args.dataset_path, args.batch_size)
    else:
        # Legacy fallback
        legacy_model_type = args.model_type or "bm25"
        legacy_dataset_path = args.dataset_path or "./jua-dataset"
        if legacy_model_type == "bm25":
            run_bm25(
                legacy_dataset_path,
                args.results_file or "results/anserini_bm25.json",
                "http://127.0.0.1:8000",
                None,
                100,
            )
        elif legacy_model_type == "dense_hf":
            run_dense_hf(args.model_name, legacy_dataset_path)
        elif legacy_model_type == "sbert":
            run_sbert(args.model_name, legacy_dataset_path, args.batch_size)
        elif legacy_model_type == "openai":
            run_openai(args.model_name, legacy_dataset_path, args.batch_size)
        elif legacy_model_type == "reranking_dense":
            run_reranking_dense(args.model_name, legacy_dataset_path, args.results_file, None)
        elif legacy_model_type == "reranking_monot5":
            run_reranking_monot5(args.model_name, legacy_dataset_path, args.batch_size)
        else:
            parser.error("Provide a valid subcommand or legacy --model_type.")
