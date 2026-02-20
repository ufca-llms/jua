from __future__ import annotations

from argparse import ArgumentParser

import jua


def main():
    parser = ArgumentParser(description="JUA benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models_parser = subparsers.add_parser("list-models", help="List registered model ids")
    list_models_parser.add_argument("--model_registry", default=None, help="Path to model registry JSON")

    list_datasets_parser = subparsers.add_parser("list-datasets", help="List registered dataset ids")
    list_datasets_parser.add_argument("--dataset_registry", default=None, help="Path to dataset registry JSON")

    run_parser = subparsers.add_parser("run", help="Run a model on a benchmark")
    run_parser.add_argument("--model", required=True, help="Model id registered in the JSON registry")
    run_parser.add_argument("--model_registry", default=None, help="Path to model registry JSON")
    run_parser.add_argument("--benchmark", default="jua", help="Benchmark id (registry) or legacy name")
    run_parser.add_argument("--dataset_registry", default=None, help="Path to dataset registry JSON")
    run_parser.add_argument("--all_datasets", action="store_true", help="Run on all registered datasets")
    run_parser.add_argument("--output_dir", default="results/leaderboard", help="Leaderboard output dir")
    run_parser.add_argument("--model_meta_json", default=None, help="Path to a JSON file with model metadata")
    run_parser.add_argument("--overall_metric", default=None, help="Overall metric (e.g. ndcg@10, mrr@10, map@10)")

    run_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    run_parser.add_argument("--max_tokens", type=int, default=3000, help="Max tokens for embeddings")
    run_parser.add_argument("--server_url", default="http://127.0.0.1:8000", help="Anserini server URL")
    run_parser.add_argument("--index_name", default=None, help="Anserini index name")
    run_parser.add_argument("--chunk_size", type=int, default=100, help="BM25 query batch size")
    run_parser.add_argument("--results_file", default=None, help="Reranking input results file")
    run_parser.add_argument("--embeddings_dir", default=None, help="Embeddings directory for reranking")

    args = parser.parse_args()

    if args.command == "list-models":
        models = jua.models.list_registered_models(args.model_registry)
        for model_id in models:
            print(model_id)
        return

    if args.command == "list-datasets":
        datasets = jua.benchmarks.list_registered_datasets(args.dataset_registry)
        for dataset_id in datasets:
            print(dataset_id)
        return

    model_meta = None
    if args.model_meta_json:
        import json
        from jua.models.model_meta import ModelMeta
        with open(args.model_meta_json, "r", encoding="utf-8") as f:
            model_meta = ModelMeta.from_dict(json.load(f))

    model = jua.get_model(
        args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        server_url=args.server_url,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        results_file=args.results_file,
        embeddings_dir=args.embeddings_dir,
        registry_path=args.model_registry,
        model_meta=model_meta,
    )

    if args.all_datasets:
        tasks = jua.get_all_benchmark_tasks(dataset_registry=args.dataset_registry)
    else:
        tasks = jua.get_tasks(args.benchmark, dataset_registry=args.dataset_registry)

    jua.run(model, tasks, output_dir=args.output_dir, overall_metric=args.overall_metric)


if __name__ == "__main__":
    main()
