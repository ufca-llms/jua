import mteb
import torch
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser


def evaluate(model_name: str,batch_size: int = 128):
    # Get the full English benchmark
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")

    # Filter to only retrieval tasks
    retrieval_tasks = mteb.filter_tasks(benchmark, task_types=["Retrieval"])
    print(f"Found {len(retrieval_tasks)} retrieval tasks")

    # Run evaluation on only retrieval tasks
    model = SentenceTransformer(model_name, model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "flash_attention_2"
    })

    model.max_seq_length = 3072

    cache = mteb.ResultCache(cache_path="~/.cache/mteb")

    results = mteb.evaluate(model, tasks=retrieval_tasks, encode_kwargs={"batch_size": batch_size}, cache=cache)
    print(results)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Model name for MTEB evaluation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for MTEB evaluation")
    args = parser.parse_args()

    evaluate(args.model_name, batch_size=args.batch_size)