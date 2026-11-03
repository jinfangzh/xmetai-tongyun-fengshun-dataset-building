"""ORAS5 → c88_p25_h24 构建示例：选择通道、归一化并准备附件。"""

import numpy as np
import xarray as xr

from datakit import io
from datakit.transforms import apply_zscore, compute_mean_std


def process(dataset, source_contract, spec, *, mean_path=None, std_path=None,
                  mask_path=None):
    """选择海洋通道 → 准备附件 → 归一化 → 组装完整产物。

    文件和契约配置由框架检查；算法由本脚本明确实现，不解释目标配方。
    """
    # 1. 读取主数据并转 float32，按名称选择、排列目标通道。保留原物理单位，不猜测历史标签映射。
    data = dataset["data"]
    data = data.astype("float32").sel(channel=list(spec.product.channel_names))

    # 2. 保留原物理单位，在当前数据域加载或计算 mean/std。
    mean, std = None, None
    if mean_path is not None:
        with io.open(mean_path).read(load=False) as source:
            mean = source["mean"].load()
    if std_path is not None:
        with io.open(std_path).read(load=False) as source:
            std = source["std"].load()
    if mean is None or std is None:
        # 只在需要计算时核对范围，避免额外日期被纳入统计量。
        if not np.array_equal(data.time.values.astype("datetime64[ns]"), spec.expected_times()):
            raise ValueError("Source time coverage must exactly match target before computing statistics")
        computed_mean, computed_std = compute_mean_std(data, dim=("time", "lat", "lon"))
        mean = computed_mean if mean is None else mean
        std = computed_std if std is None else std
    # 选择目标通道并转 float32；数值和标签合法性由 apply_zscore 检查。
    channels = list(spec.product.channel_names)
    # 允许从完整统计量中选择目标子集；选择和 float32 转换在此明确执行。
    mean = mean.sel(channel=channels).astype("float32")
    std = std.sel(channel=channels).astype("float32")

    # 3. 本用例使用外部海陆 mask，不预设内部生成方式。
    if mask_path is None:
        raise ValueError("process requires an external mask_path; implement a concrete mask recipe in this script for other sources")
    with io.open(mask_path).read(load=False) as source:
        mask = source["mask"].load()

    # 4. 只执行 zscore，不做单位转换、派生量计算或对数变换。
    data = apply_zscore(data, mean, std)

    # 5. 精确对齐附件，设置目标 dtype 与坐标，返回加工后的数组集合。
    # 转型前检查原始 mask，防止 257 等整数转换为 uint8 后伪装成合法值。
    if mask.dtype.kind not in "biu" or not np.isin(mask.values, spec.product.mask.allowed_values).all():
        raise ValueError("Mask must contain only the declared binary integer values")
    data, mean, std, mask = xr.align(data, mean, std, mask, join="exact", copy=False)
    data = data.transpose(*spec.arrays["data"].dimensions).astype(spec.arrays["data"].dtype)
    data = data.assign_coords(channel=data.channel.astype(str),
                                  time=data.time.astype("datetime64[ns]"),
                                  lat=data.lat.astype("float32"), lon=data.lon.astype("float32"))
    mean = mean.assign_coords(channel=data.channel)
    std = std.assign_coords(channel=data.channel)
    mask_coord = spec.product.mask.channel_array_name
    mask = mask.astype(spec.product.mask.dtype).assign_coords(
        {mask_coord: mask[mask_coord].astype(str), "lat": data.lat, "lon": data.lon})
    # 仅返回加工后的数组；标准属性与通道元数据由外围 pipeline 统一生成。
    return xr.Dataset({"data": data, "mean": mean, "std": std,
                       "mask": mask})


if __name__ == "__main__":
    from functools import partial

    from pipelines.dataset import build_zarr

    # 修改本次构建的契约、路径及附件；设为 True 时只加工和检查，不写出。
    dry_run = False
    report = build_zarr(
        source_config="/path/to/oras5_source.yaml",
        target_spec="/path/to/oras5_target.yaml",
        source_path="/data/oras5/input.nc",
        destination="/data/output/oras5.zarr",
        process=partial(
            process,
            mean_path=None,
            std_path=None,
            mask_path="/data/stats/oras5_mask.nc",
        ),
        dry_run=dry_run,
    )
    print(report.to_dict())
    success = (report.coverage["conformance_passed"] if dry_run
               else report.coverage["written"] and report.quick_passed)
    raise SystemExit(0 if success else 1)
