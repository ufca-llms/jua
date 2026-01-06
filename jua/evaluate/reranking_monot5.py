from beir.retrieval.evaluation import EvaluateRetrieval
from jua.models.bm25 import CustomBM25
import json
import torch
import beir
from beir.reranking import Rerank
from beir.reranking.models import MonoT5
from beir.reranking.models.mono_t5 import greedy_decode, T5BatchTokenizer
from transformers import AutoTokenizer

# Monkey patch greedy_decode to fix compatibility with newer transformers versions
def patched_greedy_decode(model, input_ids, length, attention_mask, return_last_logits):
    """Patched version that removes 'past' keyword argument and caching."""
    decode_ids = torch.full(
        (input_ids.size(0), 1),
        model.config.decoder_start_token_id,
        dtype=torch.long
    ).to(input_ids.device)
    
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=True
        )
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([
            decode_ids,
            next_token_logits.max(1)[1].unsqueeze(-1)
        ], dim=-1)
    
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids

# Monkey patch get_tokenizer to use fast tokenizer
def patched_get_tokenizer(model_path: str, *args, **kwargs) -> T5BatchTokenizer:
    """Patched version that uses fast tokenizer."""
    return T5BatchTokenizer(
        AutoTokenizer.from_pretrained(model_path, use_fast=True, *args, **kwargs)
    )

# Apply monkey patches
beir.reranking.models.mono_t5.greedy_decode = patched_greedy_decode
beir.reranking.models.mono_t5.get_tokenizer = patched_get_tokenizer

def evaluate_reranking_monot5(
        corpus: dict[str, dict[str, str]], 
        queries: dict[str, str], 
        qrels: dict[str, dict[str, str]],
        model_name: str,
        token_false: str = "_no",
        token_true: str = "_yes",
        batch_size: int = 128
    ):
    model = CustomBM25(index_path="./data/bm25_index", language="pt")
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    cross_encoder_model = MonoT5(model_name, token_false=token_false, token_true=token_true)
    reranker = Rerank(cross_encoder_model, batch_size=batch_size)

    rerank_results = reranker.rerank(corpus, queries, results, top_k=1000)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

    print(f"NDCG: {ndcg}, MAP: {_map}, Recall: {recall}, Precision: {precision}")
    
    # Remove slashes from model_name for file paths
    safe_model_name = model_name.replace("/", "_")
    
    json.dump({
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision
    }, open(f"results/{safe_model_name}_reranked_metrics.json", "w"))
    json.dump(rerank_results, open(f"results/{safe_model_name}_reranked_results.json", "w")) 
