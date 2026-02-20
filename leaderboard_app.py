from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Tuple

import gradio as gr

RESULTS_DIR = os.environ.get("JUA_LEADERBOARD_DIR", "leaderboard")
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
                    "benchmark": task_name,
                    "ndcg@10": ndcg10,
                    "generated_at": payload.get("generated_at"),
                }
            )
    return rows


def _available_benchmarks(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted({r["benchmark"] for r in rows})


def _pivot(rows: List[Dict[str, Any]], benchmarks: List[str]) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        model_id = r["model_id"]
        model_name = r["model"]
        by_model.setdefault(model_id, {"model": model_name})
        by_model[model_id][r["benchmark"]] = r.get("ndcg@10")

    pivot = []
    for model_id, data in by_model.items():
        values = [data.get(b) for b in benchmarks]
        valid = [v for v in values if isinstance(v, (int, float))]
        overall = sum(valid) / len(valid) if valid else None
        data["overall"] = overall
        pivot.append(data)
    return pivot


def build_table(selected_benchmarks: List[str], order_metric: str) -> Tuple[List[str], List[List[Any]]]:
    rows = _collect_results(RESULTS_DIR)
    all_benchmarks = _available_benchmarks(rows)

    benchmarks = selected_benchmarks if selected_benchmarks else all_benchmarks
    pivot = _pivot(rows, benchmarks)

    metric_key = order_metric
    pivot.sort(key=lambda r: (r.get(metric_key) is None, -(r.get(metric_key) or 0.0)))

    columns = ["rank", "model"] + benchmarks + ["overall"]
    data = []
    for idx, r in enumerate(pivot, start=1):
        row = [idx, r.get("model")]
        row.extend([r.get(b) for b in benchmarks])
        row.append(r.get("overall"))
        data.append(row)

    return columns, data


def _init_choices() -> List[str]:
    rows = _collect_results(RESULTS_DIR)
    return _available_benchmarks(rows)


def main():
    with gr.Blocks(title="JUA Leaderboard") as demo:
        gr.Markdown("# JUA Leaderboard")
        gr.Markdown(
            "Benchmark público para avaliação de modelos de recuperação em português, "
            "com foco em jurisprudência e normativos. "
            "Ordenação padrão por `overall` (média de NDCG@10 entre benchmarks selecionados)."
        )
        gr.Markdown(f"Results directory: `{RESULTS_DIR}`")

        with gr.Row():
            benchmark = gr.CheckboxGroup(choices=_init_choices(), value=_init_choices(), label="Benchmarks")
            order_metric = gr.Dropdown(
                choices=["overall"] + _init_choices(),
                value="overall",
                label="Order by (overall = mean ndcg@10)",
            )
            refresh = gr.Button("Refresh")

        init_cols, init_data = build_table(_init_choices(), "overall")
        table = gr.Dataframe(headers=init_cols, value=init_data, label="Ranking", interactive=False)

        def _refresh(bmks, metric):
            cols, data = build_table(bmks, metric)
            return gr.Dataframe(headers=cols, value=data, interactive=False)

        refresh.click(_refresh, inputs=[benchmark, order_metric], outputs=[table])
        benchmark.change(_refresh, inputs=[benchmark, order_metric], outputs=[table])
        order_metric.change(_refresh, inputs=[benchmark, order_metric], outputs=[table])

        gr.Markdown("---")

        logo_html = []
        brasao_uri = _img_to_data_uri(UFCA_BRASAO)
        llms_uri = _img_to_data_uri(UFCA_LLMS_LOGO)
        if brasao_uri:
            logo_html.append(f'<img src="{brasao_uri}" style="width:100px;height:auto;"/>')
        if llms_uri:
            logo_html.append(f'<img src="{llms_uri}" style="width:100px;height:auto;"/>')
        if logo_html:
            gr.HTML(
                '<div style="display:flex;gap:24px;align-items:center;justify-content:center;">'
                + "".join(logo_html)
                + "</div>"
            )

    demo.launch()


if __name__ == "__main__":
    main()
