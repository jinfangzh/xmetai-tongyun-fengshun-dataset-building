"""将后续数据 A 追加到已有数据 B，生成独立产物 C。"""

if __name__ == "__main__":
    from pipelines.dataset import append_zarr

    # 替换实际路径；C 必须不存在，目标契约描述完整 B + A。
    # 写出后内部执行 full 校验，A、B 保持不变。
    report = append_zarr(
        source="/data/era5/A.zarr",
        target="/data/era5/B.zarr",
        destination="/data/era5/C.zarr",
        spec="/path/to/era5_combined_target.yaml",
        output="/data/reports/era5_append.yaml",  # 不保存报告时设为 None
    )
    print(report.to_dict())
    raise SystemExit(0 if report.release_passed else 1)
