"""Write optional evaluation report views from finalized result tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Writer：只负责"长表之外的派生视图"
# --------------------------------------------------------------------------


def write_csv_long(tables, context, output_dir: Path) -> Path:
    """统一长表本身由 ReportWriter 无条件写出，这里只为让配置能显式声明它。"""
    return Path(output_dir) / "scores.csv"


def write_coverage(tables, context, output_dir: Path) -> Path:
    path = Path(output_dir) / "coverage.csv"
    tables.coverage.to_csv(path, index=False, encoding="utf-8")
    return path


def write_details(tables, context, output_dir: Path) -> Path:
    """列联表计数等诊断量（长表形态）。"""
    path = Path(output_dir) / "diagnostics" / "scores_detail.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tables.details.to_csv(path, index=False, encoding="utf-8")
    return path


def write_json(tables, context, output_dir: Path) -> Path:
    path = Path(output_dir) / "scores.json"
    records = json.loads(tables.score_table.to_json(orient="records", force_ascii=False))
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "run_id": context.run_id,
                "model_id": context.model_id,
                "dataset_id": context.dataset_id,
                "protocol_id": context.protocol_id,
                "scores": records,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    return path


def write_categorical_wide(tables, context, output_dir: Path) -> Path:
    from pipelines.evaluation.reporting.tables import categorical_wide

    path = Path(output_dir) / "diagnostics" / "categorical_wide.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    categorical_wide(tables).to_csv(path, index=False, encoding="utf-8")
    return path


def write_probability_wide(tables, context, output_dir: Path) -> Path:
    from pipelines.evaluation.reporting.tables import probability_wide

    path = Path(output_dir) / "diagnostics" / "probability_wide.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_wide(tables).to_csv(path, index=False, encoding="utf-8")
    return path


#: 赤道周长（km）。波长 = 赤道周长 / 纬向波数，与参考实现同一个常数。
_EARTH_CIRCUMFERENCE_KM = 40075.0


def _spectrum_number(value) -> str:
    """按参考实现的 ``%.6g`` 输出，便于跟已有归档逐格对比。"""
    return f"{float(value):.6g}"


def _spectrum_curve_mean(entries: List[Any]):
    """按样本数加权平均若干条功率谱曲线，返回 ``(pred, obs)`` 两条一维数组。

    每条曲线本来就是「该结果自身样本的平均谱」，权重取该结果的 ``n_requested``
    才等价于把所有样本一起平均（各结果的样本数不一定相同）。
    """
    import numpy as np

    size = min(
        min(
            np.asarray(curve["power_forecast"]).size,
            np.asarray(curve["power_observation"]).size,
        )
        for _, curve in entries
    )
    pred_total = np.zeros(size, dtype="f8")
    observation_total = np.zeros(size, dtype="f8")
    weight_total = 0
    for weight, curve in entries:
        keep = weight if weight > 0 else 1
        pred_total += np.asarray(curve["power_forecast"], dtype="f8")[:size] * keep
        observation_total += (
            np.asarray(curve["power_observation"], dtype="f8")[:size] * keep
        )
        weight_total += keep
    if weight_total == 0:
        return None, None
    return pred_total / weight_total, observation_total / weight_total


def write_spectrum(tables, context, output_dir: Path) -> Path:
    """逐波数功率谱：全体样本均值曲线 + 逐起报曲线。

    对标参考实现的 ``det_summary_spectrum_{var}.csv`` / ``spectrum_{init}_{var}.csv``：
    只出 k=1..720（k=0 去纬向均值后恒为 0，去掉），``wavelength_km = 40075 / k``。
    每个起报一条曲线，是其 60 个时效的平均。

    曲线不走 ``value``（那样每个波数都会展开成明细行），而是从 ``tables.curves``
    直接取原始数组。
    """
    grouped: Dict[Any, List[Any]] = {}
    for entry in tables.curves:
        curve = entry.get("curve") or {}
        if curve.get("kind") != "wavenumber_spectrum":
            continue
        base = entry.get("base") or {}
        key = (str(base.get("variable", "")), str(base.get("init_time", "")))
        grouped.setdefault(key, []).append((int(base.get("n_requested") or 0), curve))

    diagnostics = Path(output_dir) / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)

    per_variable: Dict[str, List[Any]] = {}
    by_init_rows: List[List[Any]] = []
    for (variable, init_time), entries in sorted(grouped.items()):
        per_variable.setdefault(variable, []).extend(entries)
        pred, obs = _spectrum_curve_mean(entries)
        if pred is None:
            continue
        for k in range(1, pred.size):
            by_init_rows.append(
                [
                    variable,
                    init_time,
                    k,
                    _spectrum_number(_EARTH_CIRCUMFERENCE_KM / k),
                    _spectrum_number(pred[k]),
                    _spectrum_number(obs[k]),
                ]
            )

    by_init_path = diagnostics / "spectrum_by_init.csv"
    with by_init_path.open("w", encoding="utf-8", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            ["variable", "init_time", "wavenumber", "wavelength_km", "pred", "obs"]
        )
        out.writerows(by_init_rows)

    summary_path: Optional[Path] = None
    for variable, entries in sorted(per_variable.items()):
        pred, obs = _spectrum_curve_mean(entries)
        if pred is None:
            continue
        path = diagnostics / f"spectrum_{variable}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            out = csv.writer(stream)
            out.writerow(["wavenumber", "wavelength_km", "pred_mean", "obs_mean"])
            for k in range(1, pred.size):
                out.writerow(
                    [
                        k,
                        _spectrum_number(_EARTH_CIRCUMFERENCE_KM / k),
                        _spectrum_number(pred[k]),
                        _spectrum_number(obs[k]),
                    ]
                )
        if summary_path is None:
            summary_path = path
    return summary_path or by_init_path


def _typhoon_number(value: Any) -> str:
    """逐场次 csv 的数值格式：6 位有效数字，非有限值写空。

    对齐旧归档 ``tc<编号>_<起报>.csv`` 的写法（``%.6g``）——那是给人看的表，
    6 位有效数字足够读；``str(float)`` 的 17 位尾数在这里只是噪音。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        # 非数值列（``valid_bjt`` 那种时刻字符串）原样透传：这里返回空会把
        # 整列时刻写没，而列还在，看着像"这条路径没有有效时刻"。
        return "" if value is None else str(value)
    if not math.isfinite(number):
        return ""
    return "%.6g" % number


def _typhoon_repr(value: Any) -> str:
    """拼接表 ``typhoon.csv`` 的数值格式：``str(float)`` 原样，非有限值写空。

    与逐场次表刻意不同：这张表是给下游脚本/渲染器做聚合的，取整到 6 位会
    让"同一场次在两份产物里对不上"，所以保留完整尾数。
    """
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return str(number)


def write_typhoon_cases(tables, context, output_dir: Path) -> Path:
    """台风逐场次路径表 + 拼接总表。

    对标旧归档 ``weather_typhoon_*/`` 的三个产物：

    - ``typhoon/tc<编号>_<起报>.csv``：一个场次一行一个时效，15 列，
      数值 6 位有效数字；对不上实况的时效留空；
    - ``typhoon/tc<编号>_<起报>_meta.json``：该场次的起报点、搜索参数、配对
      时效数等口径信息；
    - ``typhoon/typhoon.csv``：全部场次竖着拼起来，前面加
      ``tcid,tcname,init_utc`` 三列，数值取完整尾数。

    曲线从 ``tables.curves`` 取（``kind == "typhoon_track"``），不走 ``value``：
    15 字段 × 60 时效 × 全年场次会把长表撑爆，而曲线本来就有专门的出口。
    """
    from datakit.evaluation.metrics.track_error import CURVE_FIELDS, ENS_EXTRA_FIELDS

    cases = [
        entry
        for entry in tables.curves
        if (entry.get("curve") or {}).get("kind") == "typhoon_track"
    ]
    cases_dir = Path(output_dir) / "typhoon"
    cases_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[List[Any]] = []
    combined_fields: List[str] = list(CURVE_FIELDS)
    written: List[Path] = []
    for entry in cases:
        curve = entry["curve"]
        # 集合曲线比确定性多 5 列（方案 A 的三项 + 逐时效成员数），**确定性那条
        # 路径一列不加**：加列会让已有归档的对拍整个错位。
        is_ens = str(curve.get("forecast_type") or "") == "ens"
        fields = list(CURVE_FIELDS) + (list(ENS_EXTRA_FIELDS) if is_ens else [])
        combined_fields = fields
        tcid = str(curve.get("storm") or "")
        tcname = str(curve.get("tcname") or "")
        # 曲线里的 init_utc 是 isoformat（带 T），旧归档写的是空格分隔
        init_utc = _iso_space(curve.get("init_utc"))
        leads = list(curve.get("lead_h") or [])
        # 文件名里的起报时刻用紧凑写法（2025061000），与旧归档一致
        stamp = _compact_init(init_utc)
        stem = f"tc{tcid}_{stamp}"

        path = cases_dir / f"{stem}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            # lineterminator 必须显式给：csv 模块默认 '\r\n'，旧归档是 '\n'。
            # 不写这一行，逐字节对拍时会看到"整个文件每一行都不同"。
            out = csv.writer(stream, lineterminator="\n")
            out.writerow(fields)
            for row_index in range(len(leads)):
                out.writerow(
                    [
                        _typhoon_number(curve.get(field, [None] * len(leads))[row_index])
                        for field in fields
                    ]
                )
        written.append(path)

        matched = [
            k for k in range(len(leads)) if _is_finite(_at(curve, "track_err_km", k))
        ]
        # init_pos 取**链的种子位置**（首个能配上时效的实况，通常为首个 +6h；
        # 种子取的时刻由下面的 ``seed`` 字段标注）。首时次 at/ct 的定向基准另有
        # 一条"起报时刻实况"（ob0，见 ``TrackErrorEns``）——旧 ens 归档把
        # init_pos 也写成 ob0，是已知的记录级差异，经确认暂不追平（2026-09-30）。
        meta = {
            "tcid": tcid,
            "tcname": tcname,
            "init_utc": init_utc,
            "init_bjt": _shift_iso(init_utc, curve.get("tz_shift", 8.0)),
            "init_pos": [curve.get("init_lat"), curve.get("init_lon")],
            # 起始中心（链的种子）取自起报后哪一条实况："init+6h" 是常态（第一个
            # 时效的步长）；更大说明那条分析场缺了，比如台风当天的第一条定位在
            # 20:00，就会是 "init+12h"。和别家归档对不上 init_pos 的场次先看这个
            # 字段——曲线可以完全一样，种子取的时刻不同。
            "seed": _seed_label(curve.get("seed_offset_h")),
            "forecast_type": "ens" if is_ens else "det",
            "lead_step": _lead_step(leads),
            "tz_shift_h": curve.get("tz_shift", 8.0),
            "search": curve.get("search") or {},
            "has_wind": any(
                _is_finite(value) for value in (curve.get("fcst_vmax_ms") or [])
            ),
            "n_leads": len(leads),
            "n_matched": len(matched),
        }
        if is_ens:
            # 集合口径：曲线里的 fcst_* 与 track_err_km / at_km / ct_km 是**方案 B**
            # （集合平均位置 vs 实况），*_a 三列是**方案 A**（成员误差平均）。
            # 强度类两项两方案代数恒等，只有一列。
            meta.update(
                {
                    "n_members": len(curve.get("members") or []),
                    "members": list(curve.get("members") or []),
                    "aggregation": {
                        "B": "集合平均位置 vs 实况"
                        "（fcst_* / track_err_km / at_km / ct_km）",
                        "A": "成员误差平均（track_err_km_a / at_km_a / ct_km_a）",
                    },
                }
            )
        with (cases_dir / f"{stem}_meta.json").open("w", encoding="utf-8") as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        for row_index in range(len(leads)):
            combined_rows.append(
                [tcid, tcname, init_utc]
                + [
                    _typhoon_repr(curve.get(field, [None] * len(leads))[row_index])
                    for field in fields
                ]
            )

    combined = cases_dir / "typhoon.csv"
    with combined.open("w", encoding="utf-8", newline="") as stream:
        out = csv.writer(stream, lineterminator="\n")
        out.writerow(["tcid", "tcname", "init_utc", *combined_fields])
        out.writerows(combined_rows)
    return combined


def _seed_label(offset: Any) -> str:
    """台风 meta 的 ``seed`` 标签：起始中心取自起报后几小时的实况（"init+6h" 是常态）。"""
    try:
        hours = float(offset or 0.0)
    except (TypeError, ValueError):
        hours = 0.0
    return "init" if not hours else "init+%gh" % hours


def _at(curve: Dict[str, Any], field: str, index: int) -> Any:
    values = curve.get(field)
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return None
    return values[index]


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _iso_space(value: Any) -> str:
    """``2025-06-10T00:00:00`` -> ``2025-06-10 00:00:00``（旧归档的写法）。"""
    return str(value or "").replace("T", " ")


def _compact_init(init_utc: str) -> str:
    """``2025-06-10T00:00:00`` -> ``2025061000``（旧归档的文件名口径）。"""
    text = str(init_utc).replace("T", " ").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:10] if len(digits) >= 10 else text


def _shift_iso(init_utc: str, hours: float) -> str:
    from datetime import datetime, timedelta

    try:
        moment = datetime.fromisoformat(str(init_utc))
    except ValueError:
        return ""
    return str(moment + timedelta(hours=float(hours)))


def _lead_step(leads: List[Any]) -> Any:
    """相邻时效的间隔（单时次返回 None）。"""
    if len(leads) < 2:
        return None
    try:
        return float(leads[1]) - float(leads[0])
    except (TypeError, ValueError):
        return None


