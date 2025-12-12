from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

def evaluate_dense_hf(
    model_name: str, 
    corpus: dict[str, dict[str, str]], 
    queries: dict[str, str], 
    qrels: dict[str, dict[str, str]]):

    ## Parameters
    model_name_or_path = model_name
    max_length = 512
    pooling = "eos"
    normalize = True
    append_eos_token = True