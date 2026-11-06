"""ERA5 下载产物 → 六小时累积 → c78_p25_h6；兼容已打包物理量。"""

from contextlib import ExitStack

import numpy as np
import xarray as xr

from datakit.schema import UNIT_SCALE, load_mapping
from datakit.schema.semantics import TemporalKind
from datakit import io
from datakit.transforms import apply_log1p, apply_scale, apply_zscore
from pipelines.dataset.preparation import (
    load_external_array,
    pack_source_channels,
    prepare_standard_arrays,
    prepare_statistics,
)


# 此示例处理 6 小时累计辐射，不从文件时间间隔推断累计窗口。
ACCUMULATION_HOURS = 6.0


def read_source(source_path, description, *, spec):
    """只读取 NC/ZIP，保留每个变量独立的时间轴和文件资源。

    小时累积量和瞬时量不能先对齐到同一时间轴；使用 time_<变量> 保留
    原始时间标签，避免合并时插入 NaN 掩盖缺小时。求和与打包由 process 执行。
    """
    mapping = load_mapping(description)
    required = {channel.variable for channel in spec.product.channels}
    undeclared = sorted(required - mapping.variables.keys())
    if undeclared:
        raise ValueError(f"ERA5 source mapping is missing target variable(s): {undeclared}")
    access = io.open(source_path, description)
    records = access.discover(all_files=True)
    groups = {}
    for record in records:
        groups.setdefault(record.fields.get("group"), []).append(record)
    missing = sorted(required - groups.keys())
    if missing:
        raise FileNotFoundError(f"ERA5 download is incomplete; missing variable groups: {missing}")
    stack = ExitStack()
    try:
        prepared = []
        for variable in sorted(required):
            declaration = mapping.variables[variable]
            fields = []
            for record in groups[variable]:
                dataset = stack.enter_context(io.open(record.path).read(
                    chunks={"valid_time": 1, "time": 1, "pressure_level": 1, "level": 1},
                ))
                renames = {
                    alias: standard for alias, standard in mapping.coordinate_aliases.items()
                    if alias != standard and (alias in dataset.coords or alias in dataset.dims)
                }
                dataset = dataset.rename(renames)
                if declaration.source not in dataset.data_vars:
                    raise KeyError(f"{record.path}: missing source variable {declaration.source!r}")
                field = dataset[declaration.source].reset_coords(drop=True)
                if "time" not in field.indexes:
                    raise ValueError(f"{record.path}: expected labeled time coordinate")
                if declaration.source_unit and field.attrs.get("units") != declaration.source_unit:
                    raise ValueError(f"{record.path}: expected units {declaration.source_unit!r}, "
                                     f"observed {field.attrs.get('units')!r}")
                fields.append(field)
            value = xr.concat(fields, dim="time", join="exact", coords="minimal",
                              compat="equals", combine_attrs="identical").sortby("time")
            if not value.get_index("time").is_unique:
                raise ValueError(f"{variable}: duplicate source timestamps across monthly files")
            prepared.append(value.rename({"time": f"time_{variable}"}).to_dataset(name=declaration.source))
        raw = xr.merge(prepared, join="exact", compat="no_conflicts", combine_attrs="override")
        raw.attrs = {}
        raw.set_close(stack.close)
        return raw
    except BaseException:
        stack.close()
        raise


def _select_times(value, expected, variable):
    index = value.get_index("time")
    positions = index.get_indexer(expected.reshape(-1))
    if (positions < 0).any():
        missing = expected.reshape(-1)[positions < 0]
        raise ValueError(
            f"{variable}: incomplete source time coverage, missing {missing.size} "
            f"timestamp(s), first={missing[0]}; ending-at-time accumulation "
            "may require the preceding month's hourly files"
        )
    return positions.reshape(expected.shape)


def _prepare_raw(dataset, source_contract, spec, *, accumulation_hours):
    """ERA5 专属加工：小时量按右端六小时窗口求和，瞬时量选时，再打包。"""
    if not np.isfinite(accumulation_hours) or accumulation_hours < 1 or int(accumulation_hours) != accumulation_hours:
        raise ValueError("Raw ERA5 accumulation_hours must be a positive integer")
    hours = int(accumulation_hours)
    mapping = load_mapping(source_contract)
    expected_times = spec.expected_times()
    prepared = []
    for variable in sorted({channel.variable for channel in spec.product.channels}):
        declaration = mapping.spec_for(variable)
        if declaration is None or declaration.source not in dataset.data_vars:
            raise ValueError(f"{variable}: missing mapped ERA5 source variable")
        value = dataset[declaration.source]
        time_dim = f"time_{variable}"
        if time_dim in value.dims:
            value = value.rename({time_dim: "time"})
        if declaration.temporal_kind == TemporalKind.INTERVAL_ACCUMULATION:
            step_type = value.attrs.get("GRIB_stepType")
            if step_type is not None and step_type != "accum":
                raise ValueError(f"{variable}: expected GRIB_stepType='accum', observed {step_type!r}")
            windows = expected_times[:, None] - np.arange(hours - 1, -1, -1)[None, :] * np.timedelta64(1, "h")
            positions = _select_times(value, windows, variable)
            value = value.isel(time=xr.DataArray(positions, dims=("time", "window_hour")))
            value = value.sum("window_hour", skipna=False, keep_attrs=True)
            value = value.assign_coords(time=expected_times)
        elif declaration.temporal_kind == TemporalKind.INSTANTANEOUS:
            value = value.isel(time=_select_times(value, expected_times, variable))
        else:
            raise ValueError(f"{variable}: unsupported ERA5 temporal kind {declaration.temporal_kind.value}")
        prepared.append(value.to_dataset(name=declaration.source))
    physical = xr.merge(prepared, join="exact", compat="no_conflicts", combine_attrs="override")
    packed = pack_source_channels(physical, spec, mapping).rename("data").to_dataset()
    packed.attrs.update(accumulation_window_hours=hours, accumulation_label="right")
    return packed


def process(dataset, source_contract, spec, *, mean_path=None, std_path=None,
                  mask_path=None, accumulation_hours=ACCUMULATION_HOURS, source_stage=None):
    """核对算法前提 → 风速合成 → 单位与降水加工 → 附件准备 → 归一化 → 组装。

    文件可读性与契约配置由 pipeline 检查；这里只核对 ERA5 加工的前提。
    所有变换都使用原系数和 float32 主数据，附件生成规则保持不变。
    加工步骤由本脚本明确实现，不从目标 preprocessing 声明推导或执行算法。
    """
    # 1. 读取主数据，以 float32 执行后续加工。
    if not np.isfinite(accumulation_hours) or accumulation_hours <= 0:
        raise ValueError("accumulation_hours must be finite and positive")
    if source_stage not in (None, "raw", "packed"):
        raise ValueError("source_stage must be raw or packed")
    if source_stage == "packed" and "data" not in dataset:
        raise ValueError("packed input must contain data(time, channel, lat, lon)")
    if source_stage == "raw" and "data" in dataset:
        raise ValueError("raw input must contain unpacked ERA5 variables")
    if "data" not in dataset:
        dataset = _prepare_raw(dataset, source_contract, spec, accumulation_hours=accumulation_hours)
    observed_window = dataset.attrs.get("accumulation_window_hours")
    if observed_window is not None and observed_window != accumulation_hours:
        raise ValueError("Packed accumulation window disagrees with radiation conversion")
    data = dataset["data"].astype("float32")
    data = data.sel(time=spec.expected_times())

    # 2. 目标需要风速时，从 u/v 合成并替换源中的同名风速。
    derived = []
    for channel in spec.product.channels:
        if channel.variable in {"ws10m", "ws100m"}:
            height = "10m" if channel.variable == "ws10m" else "100m"
            wind = np.hypot(data.sel(channel=f"u{height}", drop=True),
                            data.sel(channel=f"v{height}", drop=True))
            derived.append(wind.expand_dims(channel=[channel.name]))
    if derived:
        names = [str(array.channel.values[0]) for array in derived]
        data = xr.concat([data.drop_sel(channel=[n for n in names if n in data.channel]),
                          *derived], dim="channel", join="exact")

    # 3. 选择目标通道；临时按通道拆为变量，以复用逐变量基础算子。
    data = data.sel(channel=list(spec.product.channel_names))
    names = [str(name) for name in data.channel.values]
    fields = data.to_dataset(dim="channel")

    # 4. 比湿 kg/kg → g/kg。保留旧实现按 q 前缀识别的规则。
    humidity = {name: float(UNIT_SCALE["q"]) for name in names if name.startswith("q")}
    fields = apply_scale(fields, factors=humidity)

    # 5. 辐射 J/m² → W/m²；保持原先乘以 UNIT_SCALE / 小时数的表达式。
    radiation = {"ssr", "ssrd", "fdir", "ttr"}
    scales = {name: float(UNIT_SCALE[name]) / float(accumulation_hours)
              for name in names if name in radiation}
    fields = apply_scale(fields, factors=scales)

    # 6. 保留云水等其他已定义变量的单位缩放，避免展开时遗漏旧行为。
    scales = {name: float(UNIT_SCALE[name]) for name in names
              if name in UNIT_SCALE and not name.startswith("q")
              and name not in radiation and name != "tp"}
    fields = apply_scale(fields, factors=scales)

    # 7. 降水 m → mm → 非负裁剪 → log1p，只处理存在的 tp。
    if "tp" in fields:
        fields = apply_scale(fields, factors={"tp": float(UNIT_SCALE["tp"])})
        # 裁剪是此具体降水算法的步骤，不隐藏在通用 log1p 算子中。
        fields["tp"] = fields["tp"].clip(min=0.0, keep_attrs=True)
        fields = apply_log1p(fields, variables=("tp",))
    values = fields[names].to_array(dim="channel").transpose(*data.dims)
    values = values.assign_coords(channel=data.channel)
    values.attrs, values.name = dict(data.attrs), data.name

    # 8. 在上述预处理域加载或计算 mean/std，再加载本用例要求的外部海陆 mask。
    mean, std = prepare_statistics(values, spec, mean_path=mean_path, std_path=std_path)
    mask = load_external_array(mask_path, spec.product.mask.array_name)

    # 9. 只做 zscore，禁止再次 log1p；固定 tp 统计量不匹配由 Guard 拒绝。
    values = apply_zscore(values, mean, std)

    # 精确对齐、转换目标 dtype/坐标并设置写出分块；元数据由 pipeline 组装。
    return prepare_standard_arrays(values, mean, std, mask, spec)


def main(argv=None):
    import argparse
    from functools import partial
    from pathlib import Path

    from pipelines.dataset import build_zarr
    from pipelines.dataset.reports import BuildReport, print_build_summary, save_build_report

    parser = argparse.ArgumentParser(description="ERA5 下载数据或 packed 物理量 → 标准 Zarr")
    parser.add_argument("--source-config", required=True, type=Path, help="源数据契约 YAML")
    parser.add_argument("--target-spec", required=True, type=Path, help="目标数据集契约 YAML")
    parser.add_argument("--source", required=True, type=Path, help="输入数据文件或目录")
    parser.add_argument("--destination", required=True, type=Path, help="尚不存在的目标 Zarr 目录")
    parser.add_argument("--mean-nc", type=Path, help="mean 统计量文件；省略时计算")
    parser.add_argument("--std-nc", type=Path, help="std 统计量文件；省略时计算")
    parser.add_argument("--mask-nc", required=True, type=Path, help="外部 mask 文件")
    parser.add_argument("--const-nc", type=Path, help="复制为 const.nc 的外部静态附件")
    parser.add_argument("--weight-npy", type=Path, help="复制为 weight.npy 的外部权重附件")
    parser.add_argument("--auxiliary", action="append", default=[], metavar="FILENAME=PATH",
                        help="其他契约附件；可重复指定，文件名须与目标契约一致")
    parser.add_argument("--report-dir", required=True, type=Path, help="完整 JSON 报告目录")
    parser.add_argument("--start-time", help="本次构建起始 UTC 时刻，须与 --end-time 同时提供")
    parser.add_argument("--end-time", help="本次构建结束 UTC 时刻，包含该时刻")
    parser.add_argument("--dataset-id", help="本次产物标识；覆盖时间范围时必填，建议与输出目录名一致")
    parser.add_argument("--content-version", help="本次数据内容版本；默认沿用标准契约")
    parser.add_argument("--data-revision", help="本次数据修订标识；默认沿用标准契约")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true", help="仅加工和检查（默认）")
    mode.add_argument("--write", dest="dry_run", action="store_false", help="正式写入新 Zarr 并执行 Quick 校验")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--source-stage", choices=("raw", "packed"), default="raw",
                        help="raw 聚合小时量并打包；packed 已完成上述处理（默认 raw）")
    parser.add_argument("--accumulation-hours", type=float, default=ACCUMULATION_HOURS,
                        help="累积窗口小时数，原始输入须为正整数（默认 6）")
    args = parser.parse_args(argv)
    hours = args.accumulation_hours
    if not np.isfinite(hours) or hours <= 0 or (args.source_stage == "raw" and not hours.is_integer()):
        parser.error("--accumulation-hours 必须有限且为正数；raw 输入须为整数")
    if bool(args.start_time) != bool(args.end_time):
        parser.error("--start-time 和 --end-time 必须同时提供")
    if args.start_time and not args.dataset_id:
        parser.error("指定构建时间范围时必须提供 --dataset-id")
    overrides = {key: value for key, value in {
        "first_time": args.start_time, "last_time": args.end_time, "dataset_id": args.dataset_id,
        "content_version": args.content_version, "data_revision": args.data_revision,
    }.items() if value is not None}
    auxiliaries = {}
    pairs = [("const.nc", args.const_nc), ("weight.npy", args.weight_npy)]
    for item in args.auxiliary:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("--auxiliary 格式必须为 FILENAME=PATH")
        pairs.append((name, Path(path)))
    for name, path in pairs:
        if path is None:
            continue
        if name in auxiliaries:
            parser.error(f"重复的附件参数：{name}")
        auxiliaries[name] = path
    report_dir = args.report_dir
    try:
        report = build_zarr(
            source_config=args.source_config, target_spec=args.target_spec,
            source_path=args.source, destination=args.destination,
            process=partial(process, mean_path=args.mean_nc, std_path=args.std_nc, mask_path=args.mask_nc,
                            accumulation_hours=hours, source_stage=args.source_stage),
            read=read_source if args.source_stage == "raw" else None, dry_run=args.dry_run, build_overrides=overrides, auxiliaries=auxiliaries,
        )
    except Exception as exc:
        report = BuildReport(str(args.destination.expanduser().resolve()), dry_run=args.dry_run)
        report.fail("pipeline", str(exc), exc)
    report_path = save_build_report(report, report_dir)
    print_build_summary(report, report_path, processor="era5")
    success = (report.coverage["conformance_passed"] if args.dry_run
               else report.coverage["written"] and report.quick_passed)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
