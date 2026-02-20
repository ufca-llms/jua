import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from jua.models.base import BaseModel, ModelResult
from jua.models.model_meta import ModelMeta


class RandomEmbeddings:
    def __init__(self, dim=384, seed=42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def encode_queries(self, queries, **kwargs):
        return self.rng.normal(size=(len(queries), self.dim)).astype(np.float32)

    def encode_corpus(self, corpus, **kwargs):
        return self.rng.normal(size=(len(corpus), self.dim)).astype(np.float32)


class RandomModel(BaseModel):
    def __init__(self, dim=384, seed=42, batch_size: int = 128):
        super().__init__(
            name=f"random/{dim}-{seed}",
            kind="retrieval",
            meta=ModelMeta(
                name=f"random/{dim}-{seed}",
                provider="random",
                model_type=["baseline"],
                modalities=["text"],
                description="Random embedding baseline.",
                extra={"dim": dim, "seed": seed, "batch_size": batch_size},
            ),
        )
        self.dim = dim
        self.seed = seed
        self.batch_size = batch_size

    def evaluate(self, corpus, queries, qrels, dataset_name: str, **kwargs):
        dense_model = RandomEmbeddings(dim=self.dim, seed=self.seed)
        model = DRES(dense_model, batch_size=self.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")
        embeddings_dir = kwargs.get("embeddings_dir") or f"./embeddings/{dataset_name}/random_{self.dim}_{self.seed}"
        results = retriever.encode_and_retrieve(
            corpus,
            queries,
            encode_output_path=embeddings_dir,
            overwrite=True,
        )
        # Filter results to qrels keys to avoid KeyError in custom metrics
        filtered_results = {qid: res for qid, res in results.items() if qid in qrels}

        ndcg, _map, recall, precision = retriever.evaluate(qrels, filtered_results, retriever.k_values, ignore_identical_ids=False)
        mrr = retriever.evaluate_custom(qrels, filtered_results, retriever.k_values, metric="mrr")

        metrics = {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        }
        return ModelResult(metrics=metrics, results=filtered_results)
