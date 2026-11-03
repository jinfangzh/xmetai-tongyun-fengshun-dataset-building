"""按照目标契约独立校验已有产物。"""

if __name__ == "__main__":
    from pipelines.dataset import validate_zarr

    # 替换实际路径；契约须匹配产物的完整覆盖范围。
    # 检查目标是否满足契约和声明规则，不执行源到目标的加工对照。
    mode = "full"  # 可改为 "quick" 执行快速检查
    report = validate_zarr(
        target="/data/era5/C.zarr",
        spec="/path/to/era5_combined_target.yaml",
        mode=mode,
        output="/data/reports/era5_validation.yaml",  # 不保存报告时设为 None
    )
    print(report.to_dict())
    success = report.release_passed if mode == "full" else report.quick_passed
    raise SystemExit(0 if success else 1)
