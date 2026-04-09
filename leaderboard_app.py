from __future__ import annotations

import base64
import json
import os
import html
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pandas.api.types import is_numeric_dtype

RESULTS_DIR = os.environ.get("JUA_LEADERBOARD_DIR", "leaderboard")
BENCHMARK_REGISTRY = os.environ.get("JUA_BENCHMARK_REGISTRY", "jua/benchmarks/registry.json")
ASSETS_DIR = os.environ.get("JUA_ASSETS_DIR", "assets")
UFCA_BRASAO = os.path.join(ASSETS_DIR, "ufca-brasao.png")
UFCA_LLMS_LOGO = os.path.join(ASSETS_DIR, "ufca-llms.png")


def _safe_listdir(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _benchmark_url_map() -> Dict[str, str]:
    if not os.path.exists(BENCHMARK_REGISTRY):
        return {}
    try:
        data = _load_json(BENCHMARK_REGISTRY)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}
    url_map: Dict[str, str] = {}
    for entry in data:
        name = entry.get("name")
        if not name:
            continue
        url = entry.get("url")
        if not url:
            hf_id = entry.get("hf_id")
            if hf_id:
                url = f"https://huggingface.co/datasets/{hf_id}"
        if url:
            url_map[name] = url
    return url_map


def _get_metric(metrics: Dict[str, Any], metric: str) -> float | None:
    metric = metric.lower()
    if "@" not in metric:
        return None

    name, at_k = metric.split("@", 1)
    key = name.lower()
    block = metrics.get(key) or metrics.get(key.upper())
    if isinstance(block, dict):
        return block.get(f"{name.upper()}@{at_k}") or block.get(f"{key}@{at_k}")
    return None


def _img_to_data_uri(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _collect_results(results_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for model_dir in _safe_listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        meta_path = os.path.join(model_path, "model_meta.json")
        meta = _load_json(meta_path) if os.path.exists(meta_path) else {}
        model_name = meta.get("meta", {}).get("name") or meta.get("model") or model_dir
        model_url = meta.get("meta", {}).get("url")
        model_kind = meta.get("kind")
        if not model_kind:
            meta_type = meta.get("meta", {}).get("model_type")
            if isinstance(meta_type, list):
                model_kind = "reranker" if "reranker" in meta_type else "retrieval"
            elif isinstance(meta_type, str):
                model_kind = meta_type
        if model_kind == "rerank":
            model_kind = "reranker"

        for fname in _safe_listdir(model_path):
            if not fname.endswith(".json"):
                continue
            if fname.endswith("_results.json") or fname == "model_meta.json":
                continue
            task_path = os.path.join(model_path, fname)
            payload = _load_json(task_path)
            metrics = payload.get("metrics", {})
            task_name = payload.get("task") or fname.replace(".json", "")
            ndcg10 = _get_metric(metrics, "ndcg@10")

            rows.append(
                {
                    "model_id": model_dir,
                    "model": model_name,
                    "model_url": model_url,
                    "benchmark": task_name,
                    "ndcg@10": ndcg10,
                    "generated_at": payload.get("generated_at"),
                    "model_kind": model_kind,
                }
            )
    return rows


def _available_benchmarks(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted({r["benchmark"] for r in rows})


def _display_benchmark(name: str, url_map: Dict[str, str] | None = None) -> str:
    if name.endswith("Retrieval"):
        name = name[:-9]
    name = name.replace("RFCorpus", "RF Corpus")
    if url_map:
        url = url_map.get(f"{name}Retrieval") or url_map.get(name)
        if url:
            return f"[{name}]({url})"
    return name


def _pivot(rows: List[Dict[str, Any]], benchmarks: List[str]) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        model_id = r["model_id"]
        model_name = r["model"]
        model_url = r.get("model_url")
        by_model.setdefault(model_id, {"model": model_name, "model_url": model_url})
        by_model[model_id][r["benchmark"]] = r.get("ndcg@10")

    pivot = []
    for _model_id, data in by_model.items():
        values = [data.get(b) for b in benchmarks]
        valid = [v for v in values if isinstance(v, (int, float))]
        overall = sum(valid) / len(valid) if valid else None
        data["overall"] = overall
        pivot.append(data)
    return pivot


def _get_column_widths(df: pd.DataFrame) -> list[str]:
    widths = []
    for column_name in df.columns:
        column_word_lengths = [len(word) for word in str(column_name).split()]
        if is_numeric_dtype(df[column_name]):
            value_lengths = [len(f"{value:.2f}") for value in df[column_name]]
        else:
            value_lengths = [len(str(value)) for value in df[column_name]]
        max_length = max(max(column_word_lengths), max(value_lengths))
        n_pixels = 25 + (max_length * 10)
        widths.append(f"{n_pixels}px")
    return widths


def _create_light_green_cmap():
    cmap = plt.get_cmap("Greens")
    num_colors = 256
    half_colors = np.linspace(0, 0.5, num_colors)
    half_cmap = [cmap(val) for val in half_colors]
    return LinearSegmentedColormap.from_list("LightGreens", half_cmap, N=256)


def build_table_df(selected_benchmarks: List[str], order_metric: str, kind_filter: str) -> Tuple[pd.DataFrame, List[str]]:
    rows = _collect_results(RESULTS_DIR)
    if kind_filter != "all":
        rows = [r for r in rows if r.get("model_kind") == kind_filter]
    all_benchmarks = _available_benchmarks(rows)

    benchmarks = selected_benchmarks if selected_benchmarks else all_benchmarks
    pivot = _pivot(rows, benchmarks)

    metric_key = order_metric
    pivot.sort(key=lambda r: (r.get(metric_key) is None, -(r.get(metric_key) or 0.0)))

    display_cols = ["Rank", "Model"] + [_display_benchmark(b) for b in benchmarks] + ["Overall"]
    rows_out = []
    for idx, r in enumerate(pivot, start=1):
        model_name = r.get("model", "")
        model_url = r.get("model_url")
        if model_url:
            model_cell = f"[{model_name}]({model_url})"
        else:
            model_cell = model_name
        row = [idx, model_cell]
        row.extend([r.get(b) for b in benchmarks])
        row.append(r.get("overall"))
        rows_out.append(row)

    df = pd.DataFrame(rows_out, columns=display_cols)
    numeric_cols = [c for c in df.columns if c not in ("Rank", "Model")]
    return df, numeric_cols


def _init_choices() -> List[str]:
    rows = _collect_results(RESULTS_DIR)
    return _available_benchmarks(rows)


def _build_table_component(bmks: List[str], metric: str, kind: str) -> gr.DataFrame:
    df, numeric_cols = build_table_df(bmks, metric, kind)
    cmap = _create_light_green_cmap()
    styled = df.style.format({col: "{:.4f}" for col in numeric_cols}).background_gradient(
        cmap=cmap, subset=numeric_cols
    )
    column_widths = _get_column_widths(df)
    if len(column_widths) > 0:
        column_widths[0] = "80px"
    if len(column_widths) > 1:
        column_widths[1] = "280px"

    return gr.DataFrame(
        styled,
        datatype=["number", "markdown"] + ["number"] * (len(df.columns) - 2),
        interactive=False,
        pinned_columns=2,
        column_widths=column_widths,
    )


def _benchmark_links_md(selected_benchmarks: List[str]) -> str:
    url_map = _benchmark_url_map()
    links = []
    for name in selected_benchmarks:
        label = _display_benchmark(name)
        url = url_map.get(name)
        if url:
            links.append(f"[{label}]({url})")
        else:
            links.append(label)
    if not links:
        return ""
    return "**Benchmarks:** " + " | ".join(links)


def main():
    with gr.Blocks(title="JUÁ Leaderboard") as demo:
        gr.Markdown("# JUÁ Leaderboard")
        gr.Markdown(
            "This is a public benchmark for evaluating retrieval models in Portuguese, "
            "with a focus on legal documents and regulations. "
        )
        gr.Markdown(
            "Paper: "
            "[JUÁ: An Information Retrieval Corpus of Public Audit Court Rulings](https://arxiv.org/abs/2604.06098)"
        )
        gr.Markdown(
            "Add your contribution: "
            "[model](https://github.com/ufca-llms/jua/blob/main/docs/models.md) | "
            "[benchmark](https://github.com/ufca-llms/jua/blob/main/docs/datasets.md) | "
            "[run](https://github.com/ufca-llms/jua/blob/main/docs/leaderboard.md)"
        )
        # add separator
        gr.Markdown("---")
        gr.Markdown(
            "Use the controls below to filter and sort the models based on their performance on different benchmarks. " 
            "The table shows the NDCG@10 scores for each model on the selected benchmarks, as well as an overall average score."
        )

        with gr.Row():
            benchmark = gr.CheckboxGroup(choices=_init_choices(), value=_init_choices(), label="Benchmarks")
            order_metric = gr.Dropdown(
                choices=["overall"] + _init_choices(),
                value="overall",
                label="Order by (overall = mean ndcg@10)",
            )
            kind_filter = gr.Dropdown(
                choices=["all", "retrieval", "reranker"],
                value="all",
                label="Model Type",
            )

        links_md = gr.Markdown(_benchmark_links_md(_init_choices()))
        table = _build_table_component(_init_choices(), "overall", "all")

        def _refresh(bmks, metric, kind):
            return _benchmark_links_md(bmks), _build_table_component(bmks, metric, kind)

        benchmark.change(_refresh, inputs=[benchmark, order_metric, kind_filter], outputs=[links_md, table])
        order_metric.change(_refresh, inputs=[benchmark, order_metric, kind_filter], outputs=[links_md, table])
        kind_filter.change(_refresh, inputs=[benchmark, order_metric, kind_filter], outputs=[links_md, table])

        gr.Markdown("---")

        logo_html = []
        brasao_uri = _img_to_data_uri(UFCA_BRASAO)
        llms_uri = _img_to_data_uri(UFCA_LLMS_LOGO)
        if brasao_uri:
            logo_html.append(
                f'<a href="https://ufca.edu.br" target="_blank">'
                f'<img src="{brasao_uri}" style="width:100px;height:auto;"/></a>'
            )
        if llms_uri:
            logo_html.append(
                f'<a href="https://ufca-llms.github.io" target="_blank">'
                f'<img src="{llms_uri}" style="width:100px;height:auto;"/></a>'
            )
        if logo_html:
            gr.HTML(
                '<div style="display:flex;gap:24px;align-items:center;justify-content:center;">'
                + "".join(logo_html)
                + "</div>"
            )

    demo.launch()


if __name__ == "__main__":
    main()
