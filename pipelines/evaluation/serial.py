from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from datakit.evaluation.contracts import DataIndex, DataRequest
from datakit.evaluation.errors import ConfigError, EvaluationError
from datakit.schema.semantics import TemporalKind
from pipelines.evaluation.execution.sources import (
    ClimatologyCatalog,
    ClimatologyReader,
    GriddedCatalog,
    GriddedReader,
)
from datakit.evaluation.metrics.acc import ACC
from datakit.evaluation.metrics.bias import Bias
from datakit.evaluation.metrics.ensemble import CRPS, SpreadError
from datakit.evaluation.metrics.rmse import RMSE
from datakit.evaluation.metrics.specialized import ActivityRatio, PowerSpectrum
from pipelines.evaluation.reporting import ReportWriter, RunContext
from pipelines.evaluation.components import register_builtin_components
from datakit.evaluation.ensemble import compute_ensemble_mean
from datakit.evaluation.matching import (
    DERIVED_VARIABLES,
    Matcher,
    add_derived_variables,
    narrow_batch,
)
from datakit.io.readers.layouts import CRA_LAYOUT, FENGQING_LAYOUT
from datakit.transforms.spatial import align_lon_to, align_to_grid


def run_serial_evaluation(cfg: Any) -> list:
    """从 cfg 到 scores.csv 的完整 Fengqing 网格数据流，返回指标结果列表。"""

    # 1. 细化统一配置：确定数据源、变量、指标和配对口径。
    forecast_cfg = cfg.forecast_reader
    observation_cfg = cfg.observation_reader
    reference_cfg = cfg.reference_reader or {}
    if forecast_cfg.get("type") != "fengqing" or observation_cfg.get("type") != "cra":
        raise ConfigError("此串行入口只支持 Fengqing 预报与 CRA40 实况")
    forecast_vars = _variables(forecast_cfg, "预报")
    observation_vars = _variables(observation_cfg, "实况")
    metric_options = cfg.metric_options or {}
    metrics = _build_metrics(metric_options)
    forecast_vars = _needed_variables(forecast_vars, metrics, metric_options)
    observation_vars = _needed_variables(observation_vars, metrics, metric_options)
    options = getattr(cfg, "options", None) or {}
    sample_by = str(options.get("sample_by", "valid_time"))
    if sample_by not in ("valid_time", "init_lead"):
        raise ConfigError(f"不支持的 sample_by: {sample_by}")
    if options.get("regions"):
        raise ConfigError("此 Fengqing 串行入口尚不支持 regions")
    read_leads = [
        float(value)
        for value in (forecast_cfg.get("sample_leads") or forecast_cfg.get("lead_times") or [])
    ]
    needs_reference = any(metric.needs_reference() for metric in metrics)

    # 先展开派生要素依赖，只探测所需文件；真正的数组随后按起报和时效逐块读取。
    forecast_inputs = _source_variables(forecast_vars)
    observation_inputs = _source_variables(observation_vars)
    forecast_reader, forecast_catalog = _gridded_source(forecast_cfg)
    observation_reader, observation_catalog = _gridded_source(observation_cfg)
    reference_source = (
        _climatology_source(reference_cfg) if needs_reference and reference_cfg else None
    )
    # 只发现路径和时效，不读数组；预报索引决定后续实际需要的有效时刻。
    forecast_index = forecast_catalog.discover(
        DataRequest(
            source_id=forecast_reader.source_id,
            variables=forecast_inputs,
            init_times=_candidate_init_times(cfg),
            lead_times=read_leads or None,
        )
    )
    init_times = forecast_reader.available_init_times(forecast_index)
    if cfg.limit:
        init_times = init_times[: cfg.limit]
    if not init_times:
        raise EvaluationError("评测时段内没有可用的 Fengqing 起报")
    selected_inits = set(init_times)
    records = [record for record in forecast_index.available if record.init_time in selected_inits]
    latest = _latest_init_by_valid_time(records) if sample_by == "valid_time" else {}

    # 每个 (起报, 时效) 是独立工作块；按有效时刻抽样时保留最新起报。
    chunks = []
    for init_time in init_times:
        init_records = [record for record in records if record.init_time == init_time]
        leads = sorted(
            {record.lead_time for record in init_records},
            key=lambda value: (value is None, value),
        )
        for lead in leads:
            chunk_records = [record for record in init_records if record.lead_time == lead]
            valid_time = chunk_records[0].valid_time
            if sample_by == "valid_time" and latest[valid_time] != init_time:
                continue
            chunks.append((init_time, lead, valid_time, chunk_records))
    if not chunks:
        raise EvaluationError("评测时段内没有可读取的 Fengqing 时效")

    needed_times = sorted({valid_time for _, _, valid_time, _ in chunks})
    # 对观测和参考场各做一次定向探测，避免在工作块内重复扫描目录。
    observation_index = observation_catalog.discover(
        DataRequest(
            source_id=observation_reader.source_id,
            variables=observation_inputs,
            init_times=needed_times,
        )
    )
    observation_records = {}
    for record in observation_index.available:
        observation_records.setdefault(record.init_time, []).append(record)
    missing = [time for time in needed_times if time not in observation_records]
    if missing:
        raise EvaluationError(f"缺少 {len(missing)} 个有效时刻的观测文件: {missing[:3]}")

    reference_files = {}
    if reference_source is not None:
        reference_reader, reference_catalog = reference_source
        reference_index = reference_catalog.discover(
            DataRequest(
                source_id=reference_reader.source_id,
                variables=observation_inputs,
                init_times=needed_times,
            )
        )
        reference_files = reference_index.available[0]
        missing = [time for time in needed_times if time not in reference_files]
        if missing:
            raise EvaluationError(f"缺少 {len(missing)} 个有效时刻的气候态文件: {missing[:3]}")

    states, coordinates = {}, {}
    processed = skipped = 0
    valid_times_seen, forecast_files, observation_files = set(), [], []
    forecast = observation = None
    # 只把当前工作块的数据放入内存，随即适配、配对并累计指标状态。
    for init_time, lead, valid_time, chunk_records in chunks:
        request = DataRequest(
            source_id=forecast_reader.source_id,
            variables=forecast_inputs,
            init_times=[init_time],
            lead_times=[lead] if lead is not None else None,
        )
        raw_forecast = forecast_reader.read(
            request, DataIndex(source_id=request.source_id, available=chunk_records)
        )
        raw_forecast = _prepare_variables(raw_forecast, forecast_vars)
        observation = observation_reader.read(
            DataRequest(
                source_id=observation_reader.source_id,
                variables=observation_inputs,
                init_times=[valid_time],
            ),
            DataIndex(
                source_id=observation_reader.source_id,
                available=observation_records[valid_time],
            ),
        )
        observation = _prepare_variables(observation, observation_vars)
        reference = None
        if reference_source is not None:
            reference = reference_reader.read(
                DataRequest(
                    source_id=reference_reader.source_id,
                    variables=observation_inputs,
                    init_times=[valid_time],
                ),
                DataIndex(
                    source_id=reference_reader.source_id,
                    available=[{valid_time: reference_files[valid_time]}],
                ),
            )

        forecast = compute_ensemble_mean(raw_forecast)
        batches = _pair_samples(
            forecast, observation, raw_forecast, reference, metrics, cfg,
            forecast_vars, observation_vars,
        )
        chunk_states, chunk_coordinates, chunk_processed, chunk_skipped = (
            _calculate_metrics(batches, metrics, metric_options, sample_by)
        )
        for key, values in chunk_states.items():
            states.setdefault(key, []).extend(values)
        coordinates.update(chunk_coordinates)
        processed += chunk_processed
        skipped += chunk_skipped
        valid_times_seen.update(batch.sample_keys[0]["valid_time"] for batch in batches)
        forecast_files.extend(raw_forecast.provenance.input_files)
        observation_files.extend(observation.provenance.input_files)

    if not states:
        raise EvaluationError("Fengqing 评测没有成功处理任何配对样本")
    defaults = _result_defaults(forecast_vars, observation)
    scores = _merge_results(states, coordinates, metrics, defaults)

    _write_results(
        cfg,
        scores,
        forecast,
        observation,
        sorted(valid_times_seen),
        list(dict.fromkeys(forecast_files)),
        list(dict.fromkeys(observation_files)),
        defaults,
        processed,
        skipped,
    )
    return scores


def _variables(source_cfg: dict, label: str) -> list[str]:
    names = source_cfg.get("variables") or (
        [source_cfg["variable"]] if source_cfg.get("variable") else []
    )
    if not names:
        raise ConfigError(f"{label}数据源需要 variable 或 variables")
    return [str(name) for name in names]


def _parse_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y%m%d%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)  # noqa: DTZ007 - 文件时刻没有时区
        except ValueError:
            pass
    raise ConfigError(f"无效日期: {value}")


def _source_variables(targets: list[str]) -> list[str]:
    """把目标派生要素展开为必须从文件读取的源要素。"""
    names = []
    for target in targets:
        for name in DERIVED_VARIABLES.get(target, (target,)):
            if name not in names:
                names.append(name)
    return names


def _needed_variables(configured: list[str], metrics: list, options: dict) -> list[str]:
    """只保留指标要用的目标要素；未限定要素的指标需要配置中的全部要素。"""
    requested = set()
    for metric in metrics:
        names = options[metric.name].get("variables")
        if not names:
            return configured
        requested.update(names)
    selected = [name for name in configured if name in requested]
    if not selected:
        raise ConfigError(f"指标要素 {sorted(requested)} 不在数据源配置要素 {configured} 中")
    return selected


def _candidate_init_times(cfg: Any) -> list[datetime]:
    """配置中的候选起报；是否有文件由 Catalog 决定。"""
    source = cfg.forecast_reader
    times = [datetime.fromisoformat(str(value)) for value in source.get("init_times") or []]
    if times:
        return sorted(times)
    start = _parse_date(cfg.start_date)
    end = _parse_date(cfg.end_date or cfg.start_date)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _gridded_source(source: dict):
    source_type = source["type"]
    root = source.get("root_dir") or source.get("root")
    if not root:
        raise ConfigError(f"{source_type} 数据源需要 root_dir")
    if source_type == "fengqing":
        step = source.get("step_hours")
        layout = FENGQING_LAYOUT.with_step_hours(float(step)) if step is not None else FENGQING_LAYOUT
    else:
        layout = CRA_LAYOUT
    source_id = source.get("source_id", source_type)
    return GriddedReader(source_id=source_id, layout=layout), GriddedCatalog(Path(root), layout)


def _latest_init_by_valid_time(records: list) -> dict:
    latest = {}
    for record in records:
        if record.valid_time is not None:
            latest[record.valid_time] = max(record.init_time, latest.get(record.valid_time, record.init_time))
    return latest


def _climatology_source(source: dict):
    """构造参考场读取句柄，具体时刻由执行循环给出。"""
    if source.get("type") != "climatology":
        raise ConfigError("此串行入口只支持 climatology 参考场")
    root = source.get("root_dir") or source.get("root")
    if not root:
        raise ConfigError("climatology 数据源需要 root_dir")
    source_id = source.get("source_id", "climatology")
    reader = ClimatologyReader(source_id=source_id, engine=str(source.get("engine", "cfgrib")))
    catalog = ClimatologyCatalog(Path(root))
    return reader, catalog


def _prepare_variables(bundle, variables: list[str]):
    """逐源计算请求的派生要素，并补齐其语义；集合降维在此之后进行。"""
    before = set(bundle.payload.data_vars)
    payload = add_derived_variables(bundle.payload, variables)
    added = set(payload.data_vars) - before
    if not added:
        return bundle
    temporal = bundle.semantic.temporal_kind
    if isinstance(temporal, dict):
        temporal = {**temporal, **{name: TemporalKind.INSTANTANEOUS for name in added}}
    semantic = replace(
        bundle.semantic,
        units={**bundle.semantic.units, **{name: "m/s" for name in added}},
        temporal_kind=temporal,
    )
    provenance = replace(
        bundle.provenance,
        transform_chain=[*bundle.provenance.transform_chain, *[f"derive_{name}" for name in sorted(added)]],
    )
    return replace(bundle, payload=payload, semantic=semantic, provenance=provenance)


def _pair_samples(
    forecast, observation, raw_forecast, reference, metrics: list, cfg: Any,
    forecast_vars: list[str], observation_vars: list[str],
):
    """配对预报和实况，并为需要的指标附上成员场和气候态。"""
    options = getattr(cfg, "options", None) or {}
    sample_by = str(options.get("sample_by", "valid_time"))
    member_vars = _member_variables(metrics, cfg.metric_options or {}, forecast_vars)
    batches = Matcher(ensemble_reduction=str(options.get("ensemble_reduction", "mean"))).match(
        forecast, observation, variables=forecast_vars, sample_by=sample_by
    )
    batches.sort(
        key=lambda batch: (
            batch.sample_keys[0].get("valid_time", ""),
            batch.sample_keys[0].get("init_time", ""),
        )
    )
    for batch in batches:
        if member_vars:
            batch.members = _members_for(batch, raw_forecast, member_vars)
        if reference is not None:
            batch.reference = _reference_for(batch, reference, observation_vars)
    return batches


def _calculate_metrics(batches: list, metrics: list, metric_options: dict, sample_by: str):
    """逐配对样本计算指标，只收集可合并的统计状态。"""
    states = {}
    coordinates = {}
    processed = skipped = 0
    for batch in batches:
        record = batch.sample_keys[0]
        sample_coordinates = {
            "valid_time": record.get("valid_time", ""),
            "init_time": record.get("init_time", ""),
            "lead_h": record.get("lead_h", ""),
            "sample_unit": batch.sample_dim,
        }
        sample_key = (
            (
                sample_coordinates["valid_time"],
                sample_coordinates["init_time"],
                sample_coordinates["lead_h"],
            )
            if sample_by == "init_lead"
            else sample_coordinates["valid_time"]
        )
        calculated = False
        for index, metric in enumerate(metrics):
            for variable in metric_options[metric.name].get("variables") or [None]:
                target = narrow_batch(batch, variable) if variable else batch
                if target is None:
                    continue
                metric.validate(target)
                state = metric.accumulate(target)
                key = (index, sample_key, variable)
                states.setdefault(key, []).append(state)
                coordinates.setdefault(
                    key,
                    {
                        **sample_coordinates,
                        **({"variable": variable} if variable else {}),
                    },
                )
                calculated = True
        if calculated:
            processed += 1
        else:
            skipped += 1
    return states, coordinates, processed, skipped


def _result_defaults(forecast_vars: list[str], observation) -> dict:
    """为单变量结果补齐变量、样本类型和单位。"""
    single = len(forecast_vars) == 1
    return {
        "variable": forecast_vars[0] if single else "",
        "sample_unit": "grid",
        "unit": str(observation.semantic.units.get(forecast_vars[0], "")) if single else "",
    }


def _merge_results(states: dict, coordinates: dict, metrics: list, defaults: dict) -> list:
    """合并相同结果键的状态，再计算最终分数和结果坐标。"""
    scores = []
    for key, values in states.items():
        metric = metrics[key[0]]
        score = metric.finalize(metric.merge(values))
        score.coordinates = {**defaults, **coordinates[key]}
        score.product_kind = score.product_kind or getattr(metric, "PRODUCT_KIND", "")
        score.unit = score.unit or defaults["unit"]
        score.protocol_id = score.protocol_id or "grid_valid_time"
        scores.append(score)
    return scores


def _write_results(
    cfg: Any,
    scores: list,
    forecast,
    observation,
    valid_times: list[str],
    forecast_files: list[str],
    observation_files: list[str],
    defaults: dict,
    processed: int,
    skipped: int,
) -> None:
    """统一写出指标结果和运行记录。"""
    forecast_id = cfg.forecast_reader.get("source_id", "fengqing")
    observation_id = cfg.observation_reader.get("source_id", "cra")
    manifest = {
        "description": cfg.description,
        "n_processed_batches": processed,
        "n_skipped_batches": skipped,
        "forecast_source": forecast_id,
        "observation_source": observation_id,
        "execution": {"mode": "serial", "chunking": "init_lead"},
        "valid_times": valid_times,
        "forecast_input_files": forecast_files,
        "observation_input_files": observation_files,
        "forecast_reader": f"{forecast.provenance.reader_id}@{forecast.provenance.reader_version}",
        "observation_reader": (
            f"{observation.provenance.reader_id}@{observation.provenance.reader_version}"
        ),
    }
    register_builtin_components()
    ReportWriter(
        Path(cfg.output_dir),
        RunContext(cfg.name, forecast_id, observation_id, "grid_valid_time"),
        defaults=defaults,
    ).write(
        scores,
        manifest=manifest,
        resolved_config=asdict(cfg),
        writers=(
            cfg.writers
            if cfg.writers is not None
            else (["csv_long", "spectrum"] if "spectrum" in cfg.metric_options else ["csv_long"])
        ),
    )


def _build_metrics(options: dict) -> list:
    """cfg.metric_options 明确指定要运行的指标及其参数。"""
    constructors = {
        "rmse": lambda p: RMSE(p),
        "bias": lambda p: Bias(p),
        "acc": lambda p: ACC({"climatology_path": None, **p}),
        "activity": lambda p: ActivityRatio(p),
        "crps": lambda p: CRPS(p),
        "spread_error": lambda p: SpreadError(ddof=int(p.pop("ddof", 1)), params=p),
        "spectrum": lambda p: PowerSpectrum(
            max_wavenumber=int(p.pop("max_wavenumber", 30)),
            lat_order=p.pop("lat_order", None),
            params=p,
        ),
    }
    if not options:
        raise ConfigError("Fengqing 串行评测需要在 metric_options 中列出要算的指标")
    unknown = set(options) - set(constructors)
    if unknown:
        raise ConfigError(f"Fengqing 串行评测不支持指标: {sorted(unknown)}")
    return [constructors[name](dict(params or {})) for name, params in options.items()]


def _member_variables(metrics: list, metric_options: dict, forecast_vars: list[str]) -> list[str]:
    names = []
    for metric in metrics:
        if metric.needs_members():
            for name in metric_options[metric.name].get("variables") or forecast_vars:
                if name not in names:
                    names.append(name)
    return names


def _members_for(batch, raw_forecast, variables: list[str]):
    record = batch.sample_keys[0]
    payload = raw_forecast.payload
    available = [name for name in variables if name in payload.data_vars]
    if not available or "member" not in payload.dims:
        return None
    try:
        field = payload[available].sel(
            init_time=np.datetime64(record["init_time"]),
            lead_time=float(record["lead_h"]),
        )
    except (KeyError, ValueError):
        return None
    if "member" not in field.dims:
        return None
    target = batch.observation
    aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
    return aligned if aligned is not None else field.interp(lat=target["lat"], lon=target["lon"])


def _reference_for(batch, reference, variables: list[str]):
    record = batch.sample_keys[0]
    available = [name for name in _source_variables(variables) if name in reference.payload.data_vars]
    if not available:
        return None
    try:
        field = reference.payload[available].sel(valid_time=np.datetime64(record["valid_time"]))
    except (KeyError, ValueError):
        return None
    target = batch.observation
    field = add_derived_variables(align_lon_to(field, target["lon"].values), variables)
    field = field[[name for name in variables if name in field.data_vars]]
    aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
    return aligned if aligned is not None else field.interp(lat=target["lat"], lon=target["lon"])
