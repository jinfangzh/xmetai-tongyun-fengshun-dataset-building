"""报告使用的运行信息和区域标签，直接读取产物中的声明。"""

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def manifest_info_from(manifest: dict[str, object] | None) -> dict[str, object]:
    """把 ``manifest.json`` 的原始 dict 翻成报告要用的字段。

    只搬运，不加工：拿不到的一律留空，由渲染层如实写"未提供"。
    """
    if not manifest:
        return {}
    resolved = manifest.get("resolved_config") or {}
    forecast = resolved.get("forecast") or {} if isinstance(resolved, dict) else {}
    observation = (
        resolved.get("observation") or {} if isinstance(resolved, dict) else {}
    )
    reference = resolved.get("reference") or {} if isinstance(resolved, dict) else {}
    return {
        "run_id": manifest.get("run_id"),
        "pipeline": resolved.get("pipeline") if isinstance(resolved, dict) else None,
        "protocol_id": manifest.get("protocol_id"),
        "description": manifest.get("description"),
        "model_id": manifest.get("forecast_source") or manifest.get("model_id"),
        "dataset_id": manifest.get("observation_source") or manifest.get("dataset_id"),
        "forecast_reader": forecast.get("reader")
        if isinstance(forecast, dict)
        else None,
        "observation_reader": observation.get("reader")
        if isinstance(observation, dict)
        else None,
        "reference_reader": reference.get("reader")
        if isinstance(reference, dict)
        else None,
        "start_date": resolved.get("start_date")
        if isinstance(resolved, dict)
        else None,
        "end_date": resolved.get("end_date") if isinstance(resolved, dict) else None,
        "periods": resolved.get("periods") if isinstance(resolved, dict) else None,
        "created_at": manifest.get("created_at"),
        "n_results": manifest.get("n_results"),
        "scores_rows": manifest.get("scores_rows"),
        "details_rows": manifest.get("details_rows"),
        "statuses": manifest.get("statuses"),
        "metric_options": resolved.get("metric_options")
        if isinstance(resolved, dict)
        else None,
        "forecast_source_files": list(manifest.get("forecast_input_files") or []),
        "observation_source_files": list(manifest.get("observation_input_files") or []),
    }


def requested_from(info: dict[str, object]) -> dict[str, list[str]]:
    """从 ``metric_options`` 取"每个指标要求算哪些变量"。"""
    options = info.get("metric_options") or {}
    if not isinstance(options, dict):
        return {}
    requested: dict[str, list[str]] = {}
    for metric, params in options.items():
        if isinstance(params, dict) and params.get("variables"):
            requested[str(metric)] = [str(name) for name in params["variables"]]
    return requested


def split_regions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """宽表 → ``(全球行, {带名: 该带的行})``。

    分纬度带评测时同一张宽表里全球行与各带行并存（全球行的 ``region`` 空），
    **两者绝不能混着算**：各带平均再平均 ≠ 全球平均，混进去会把带间差异整个
    抹平。所以图与正文一律只看全球行，带行走 :func:`ts_report.build_ts_report`
    的「附 L」那一块。

    没有 ``region`` 列（这个能力出现之前的产物、或没配分带）时：全部算全球行、
    带集为空——**与加这个能力之前逐行一致**。函数本身是幂等的，传进来的已经是
    全球行时得到的还是它自己。
    """
    if "region" not in df.columns:
        return df, {}
    values = df["region"]
    empty = values.isna() | (values.astype(str).str.strip() == "")
    if bool(empty.all()):
        return df, {}
    bands = {
        name: df[~empty & (values.astype(str).str.strip() == name)]
        for name in sorted(set(values[~empty].astype(str).str.strip()))
    }
    return df[empty], bands


_REGION_ENTRY = re.compile(r"^(?P<name>.+?)\s*\[(?P<bounds>.+)\]$")


def region_labels_from(root: Path) -> dict[str, str]:
    """读取配置声明的区域边界；无文件或无有效 JSON 时返回空标签。"""
    try:
        manifest = json.loads(
            (Path(root) / "manifest.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ValueError):
        return {}
    labels = {}
    for entry in manifest.get("regions") or []:
        match = _REGION_ENTRY.match(str(entry).strip())
        if match:
            labels[match.group("name").strip()] = match.group("bounds").strip()
    return labels


def region_display(name: str, labels: Mapping[str, str]) -> str:
    """区域名附上已声明的边界；不推断缺失边界。"""
    bounds = labels.get(name)
    return f"{name} [{bounds}]" if bounds else str(name)
