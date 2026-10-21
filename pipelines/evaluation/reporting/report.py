"""Generate a basic Markdown report and plots from finalized evaluation scores."""

from __future__ import annotations

import json
from pathlib import Path


def build_report(
    output_dir: Path,
    *,
    out_dir: Path | None = None,
    model_name: str | None = None,
) -> dict[str, Path]:
    """Read stored scores without recomputing metrics; return generated files.

    Curves keep product, region, threshold, group, unit and aggregation separate.
    Values are shown per initialization; no new cross-sample average is inferred.
    This is a basic results report, not the legacy multi-model diagnostic report.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    scores_path = output_dir / "scores.csv"
    data = pd.read_csv(scores_path, low_memory=False)
    required = {"variable", "metric", "lead_h", "value", "status"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"scores.csv 缺少报告必需列：{', '.join(sorted(missing))}")
    manifest_path = output_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    # Import plotting only when a report is requested, before creating artifacts.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    target = Path(out_dir) if out_dir is not None else output_dir / "report"
    target.mkdir(parents=True, exist_ok=True)
    title = model_name or manifest.get("run_id") or output_dir.name
    lines = [
        f"# {title} 评测结果报告", "",
        f"评测结果目录：`{output_dir.resolve()}`", "",
        f"评分记录：{len(data)} 行。", "",
        "状态统计：" + ", ".join(
            f"{name}={count}" for name, count in data["status"].fillna("unknown").value_counts().items()
        ), "",
        "曲线按起报分别展示已保存的评分，不重新计算指标，也不跨区域、阈值或产品类型求平均。",
        "谱指标仅展示已保存的标量；本报告不推断谱类型、波长或频带。", "",
    ]
    artifacts: dict[str, Path] = {}
    # Include every available scoring dimension except time and sample counts.
    dimensions = [name for name in (
        "run_id", "model_id", "dataset_id", "variable", "level", "metric",
        "product_kind", "region", "group", "threshold", "window_h", "unit",
        "aggregation", "weights_id", "sample_unit", "storm", "protocol_id",
        "metric_version",
    ) if name in data.columns]
    usable = data.loc[data["status"].eq("success")].copy()
    usable["value"] = pd.to_numeric(usable["value"], errors="coerce")
    usable["lead_h"] = pd.to_numeric(usable["lead_h"], errors="coerce")
    usable = usable.loc[usable["value"].notna() & usable["lead_h"].notna()]
    for index, (keys, rows) in enumerate(usable.groupby(dimensions, dropna=False, sort=False), 1):
        labels = [f"{name}={value}" for name, value in zip(dimensions, keys)
                  if pd.notna(value) and str(value) != ""]
        lines.extend([f"## 评分组 {index}", "", " · ".join(labels), ""])
        figure = Figure(figsize=(8, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        series_columns = [name for name in ("init_time", "valid_time") if name in rows.columns]
        # Initialization defines a curve; valid time is retained in the table.
        series = rows.groupby("init_time", dropna=False, sort=False) if "init_time" in rows else [("scores", rows)]
        for name, frame in series:
            frame = frame.sort_values("lead_h")
            if frame["lead_h"].duplicated().any():
                axes.scatter(frame["lead_h"], frame["value"], label=str(name))
            else:
                axes.plot(frame["lead_h"], frame["value"], marker="o", label=str(name))
        axes.set_xlabel("Lead time (h)")
        axes.set_ylabel(str(rows.iloc[0]["metric"]))
        axes.grid(alpha=0.3)
        if "init_time" in rows and rows["init_time"].nunique(dropna=False) <= 12:
            axes.legend(fontsize="small")
        path = target / f"scores_{index:03d}.png"
        figure.savefig(path, dpi=120)
        artifacts[f"plot_{index}"] = path
        lines.extend([f"![评分曲线]({path.name})", ""])
        columns = series_columns + ["lead_h", "value"]
        lines.extend(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |"])
        for values in rows.sort_values(series_columns + ["lead_h"])[columns].itertuples(index=False, name=None):
            cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if usable.empty:
        lines.extend(["没有可绘图的成功评分记录，请检查评测日志和输入资料。", ""])
    report_path = target / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts["report"] = report_path
    return artifacts
