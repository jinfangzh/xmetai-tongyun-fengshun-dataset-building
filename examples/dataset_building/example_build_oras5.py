"""ORAS5 → c88_p25_h24 构建示例：选择通道、归一化并准备附件。"""

from datakit.transforms import apply_zscore
from pipelines.dataset.preparation import (
    load_external_array,
    pack_source_channels,
    prepare_standard_arrays,
    prepare_statistics,
)


def process(dataset, source_contract, spec, *, mean_path=None, std_path=None,
                  mask_path=None):
    """选择海洋通道 → 准备附件 → 归一化 → 组装完整产物。

    文件和契约配置由框架检查；算法由本脚本明确实现，不解释目标配方。
    """
    # 1. 读取主数据并转 float32，按名称选择、排列目标通道。保留原物理单位，不猜测历史标签映射。
    data = (dataset["data"] if "data" in dataset
            else pack_source_channels(dataset, spec, source_contract))
    data = data.sel(time=spec.expected_times())
    data = data.astype("float32").sel(channel=list(spec.product.channel_names))

    # 2. 保留原物理单位，在当前数据域加载或计算 mean/std。
    mean, std = prepare_statistics(data, spec, mean_path=mean_path, std_path=std_path)
    mask = load_external_array(mask_path, spec.product.mask.array_name)

    # 4. 只执行 zscore，不做单位转换、派生量计算或对数变换。
    data = apply_zscore(data, mean, std)

    # 精确对齐、转换目标 dtype/坐标并设置写出分块；元数据由 pipeline 组装。
    return prepare_standard_arrays(data, mean, std, mask, spec)


def main(argv=None):
    import argparse
    from functools import partial
    from pathlib import Path

    from pipelines.dataset import build_zarr
    from pipelines.dataset.reports import BuildReport, print_build_summary, save_build_report

    parser = argparse.ArgumentParser(description="ORAS5 日文件或 packed 物理量 → 标准 Zarr")
    parser.add_argument("--source-config", required=True, type=Path, help="源数据契约 YAML")
    parser.add_argument("--target-spec", required=True, type=Path, help="目标数据集契约 YAML")
    parser.add_argument("--source", required=True, type=Path, help="输入数据文件或目录")
    parser.add_argument("--destination", required=True, type=Path, help="尚不存在的目标 Zarr 目录")
    parser.add_argument("--mean-nc", type=Path, help="mean 统计量文件；省略时计算")
    parser.add_argument("--std-nc", type=Path, help="std 统计量文件；省略时计算")
    parser.add_argument("--mask-nc", required=True, type=Path, help="外部 mask 文件")
    parser.add_argument("--const-nc", type=Path, help="复制为 const.nc 的外部静态附件")
    parser.add_argument("--weight-nc", type=Path, help="复制为 weight.nc 的外部权重附件")
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
    args = parser.parse_args(argv)
    if bool(args.start_time) != bool(args.end_time):
        parser.error("--start-time 和 --end-time 必须同时提供")
    if args.start_time and not args.dataset_id:
        parser.error("指定构建时间范围时必须提供 --dataset-id")
    overrides = {key: value for key, value in {
        "first_time": args.start_time, "last_time": args.end_time, "dataset_id": args.dataset_id,
        "content_version": args.content_version, "data_revision": args.data_revision,
    }.items() if value is not None}
    auxiliaries = {}
    pairs = [("const.nc", args.const_nc), ("weight.nc", args.weight_nc)]
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
            process=partial(process, mean_path=args.mean_nc, std_path=args.std_nc, mask_path=args.mask_nc),
            dry_run=args.dry_run, build_overrides=overrides, auxiliaries=auxiliaries,
        )
    except Exception as exc:
        report = BuildReport(str(args.destination.expanduser().resolve()), dry_run=args.dry_run)
        report.fail("pipeline", str(exc), exc)
    report_path = save_build_report(report, report_dir)
    print_build_summary(report, report_path, processor="oras5")
    success = (report.coverage["conformance_passed"] if args.dry_run
               else report.coverage["written"] and report.quick_passed)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
