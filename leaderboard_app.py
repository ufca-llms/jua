from __future__ import annotations

import base64
import json
import os
import html
from typing import Any, Dict, List

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


def _display_benchmark(name: str) -> str:
    if name.endswith("Retrieval"):
        name = name[:-9]
    name = name.replace("RFCorpus", "RF Corpus")
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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _column_ranges(rows: List[Dict[str, Any]], benchmarks: List[str]) -> Dict[str, tuple[float, float]]:
    ranges: Dict[str, tuple[float, float]] = {}
    keys = benchmarks + ["overall"]
    for k in keys:
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        ranges[k] = (min(vals), max(vals))
    return ranges


def _heat_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 1.0
    else:
        t = (value - vmin) / (vmax - vmin)
    # purple to green scale (like mteb-ish)
    r1, g1, b1 = (88, 54, 158)
    r2, g2, b2 = (46, 163, 102)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def build_table_html(selected_benchmarks: List[str], order_metric: str, kind_filter: str) -> str:
    rows = _collect_results(RESULTS_DIR)
    if kind_filter != "all":
        rows = [r for r in rows if r.get("model_kind") == kind_filter]
    all_benchmarks = _available_benchmarks(rows)

    benchmarks = selected_benchmarks if selected_benchmarks else all_benchmarks
    pivot = _pivot(rows, benchmarks)

    metric_key = order_metric
    pivot.sort(key=lambda r: (r.get(metric_key) is None, -(r.get(metric_key) or 0.0)))

    ranges = _column_ranges(pivot, benchmarks)

    headers = ["rank", "model"] + [_display_benchmark(b) for b in benchmarks] + ["overall"]

    html_rows = []
    for idx, r in enumerate(pivot, start=1):
        model_name = html.escape(r.get("model", ""))
        model_url = r.get("model_url")
        if model_url:
            model_cell = f'<a href="{html.escape(model_url)}" target="_blank">{model_name}</a>'
        else:
            model_cell = model_name

        cells = [str(idx), model_cell]
        for b in benchmarks:
            val = r.get(b)
            if isinstance(val, (int, float)) and b in ranges:
                vmin, vmax = ranges[b]
                bg = _heat_color(float(val), vmin, vmax)
                cells.append(f'<span class="num-cell" style="background:{bg};">{_fmt(val)}</span>')
            else:
                cells.append(_fmt(val))
        overall_val = r.get("overall")
        if isinstance(overall_val, (int, float)) and "overall" in ranges:
            vmin, vmax = ranges["overall"]
            bg = _heat_color(float(overall_val), vmin, vmax)
            cells.append(f'<span class="num-cell" style="background:{bg};">{_fmt(overall_val)}</span>')
        else:
            cells.append(_fmt(overall_val))

        html_cells = "".join([f"<td>{c}</td>" for c in cells])
        html_rows.append(f"<tr>{html_cells}</tr>")

    header_html = "".join([f"<th>{html.escape(h)}</th>" for h in headers])
    body_html = "".join(html_rows)

    return f"""
    <style>
      .lb-wrap {{
        max-height: 520px;
        overflow: auto;
        border: 1px solid #2a2f3a;
        border-radius: 14px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
      }}
      .lb-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
      }}
      .lb-table th {{
        position: sticky;
        top: 0;
        background: linear-gradient(180deg, #171a21 0%, #12151b 100%);
        color: #e9edf3;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.08em;
        border-bottom: 1px solid #2a2f3a;
        padding: 10px 12px;
        z-index: 2;
        white-space: nowrap;
      }}
      .lb-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid #1e232b;
        color: #d7dbe3;
      }}
      .lb-table tr:nth-child(even) td {{
        background: #0f1217;
      }}
      .lb-table tr:hover td {{
        background: #141922;
      }}
      .lb-table td:first-child,
      .lb-table th:first-child {{
        position: sticky;
        left: 0;
        background: #0f1116;
        z-index: 3;
      }}
      .lb-table td:nth-child(2),
      .lb-table th:nth-child(2) {{
        position: sticky;
        left: 48px;
        background: #0f1116;
        z-index: 3;
        min-width: 240px;
      }}
      .lb-table a {{
        color: #9ecbff;
        text-decoration: none;
      }}
      .lb-table a:hover {{
        text-decoration: underline;
      }}
      .num-cell {{
        display: inline-block;
        min-width: 64px;
        padding: 4px 6px;
        border-radius: 6px;
        color: #fff;
        text-align: center;
        font-variant-numeric: tabular-nums;
      }}
    </style>
    <div class="lb-wrap">
      <table class="lb-table">
        <thead>
          <tr>{header_html}</tr>
        </thead>
        <tbody>
          {body_html}
        </tbody>
      </table>
    </div>
    """


def _init_choices() -> List[str]:
    rows = _collect_results(RESULTS_DIR)
    return _available_benchmarks(rows)


def main():
    with gr.Blocks(title="JUA Leaderboard") as demo:
        gr.Markdown("# JUA Leaderboard")
        gr.Markdown(
            "Public benchmark for evaluating retrieval models in Portuguese"
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

        table_html = gr.HTML(build_table_html(_init_choices(), "overall", "all"))

        def _refresh(bmks, metric, kind):
            return build_table_html(bmks, metric, kind)

        benchmark.change(_refresh, inputs=[benchmark, order_metric, kind_filter], outputs=[table_html])
        order_metric.change(_refresh, inputs=[benchmark, order_metric, kind_filter], outputs=[table_html])
        kind_filter.change(_refresh, inputs=[benchmark, order_metric, kind_filter], outputs=[table_html])

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
