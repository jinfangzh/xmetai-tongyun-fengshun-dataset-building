"""ERA5 → c78_p25_h6 构建示例：加工 packed 数据并返回完整 Dataset。"""

import numpy as np
import xarray as xr

from datakit.schema import UNIT_SCALE
from datakit import io
from datakit.transforms import apply_log1p, apply_scale, apply_zscore, compute_mean_std


# 此示例处理 6 小时累计辐射，不从文件时间间隔推断累计窗口。
ACCUMULATION_HOURS = 6.0


def process(dataset, source_contract, spec, *, mean_path=None, std_path=None,
                  mask_path=None):
    """核对算法前提 → 风速合成 → 单位与降水加工 → 附件准备 → 归一化 → 组装。

    文件可读性与契约配置由 pipeline 检查；这里只核对 ERA5 加工的前提。
    所有变换都使用原系数和 float32 主数据，附件生成规则保持不变。
    加工步骤由本脚本明确实现，不从目标 preprocessing 声明推导或执行算法。
    """
    # 1. 读取主数据，以 float32 执行后续加工。
    data = dataset["data"].astype("float32")

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
    scales = {name: float(UNIT_SCALE[name]) / float(ACCUMULATION_HOURS)
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
    mean, std = None, None
    if mean_path is not None:
        with io.open(mean_path).read(load=False) as source:
            mean = source["mean"].load()
    if std_path is not None:
        with io.open(std_path).read(load=False) as source:
            std = source["std"].load()
    if mean is None or std is None:
        # 只在需要计算时核对范围，避免额外日期被纳入统计量。
        if not np.array_equal(values.time.values.astype("datetime64[ns]"), spec.expected_times()):
            raise ValueError("Source time coverage must exactly match target before computing statistics")
        computed_mean, computed_std = compute_mean_std(values, dim=("time", "lat", "lon"))
        mean = computed_mean if mean is None else mean
        std = computed_std if std is None else std
    # 选择目标通道并转 float32；数值和标签合法性由 apply_zscore 检查。
    channels = list(spec.product.channel_names)
    # 允许从完整统计量中选择目标子集；选择和 float32 转换在此明确执行。
    mean = mean.sel(channel=channels).astype("float32")
    std = std.sel(channel=channels).astype("float32")
    if mask_path is None:
        raise ValueError("process requires an external mask_path; implement a concrete mask recipe in this script for other sources")
    with io.open(mask_path).read(load=False) as source:
        mask = source["mask"].load()

    # 9. 只做 zscore，禁止再次 log1p；固定 tp 统计量不匹配由 Guard 拒绝。
    values = apply_zscore(values, mean, std)

    # 10. 精确对齐附件，设置目标 dtype 与坐标，返回加工后的数组集合。
    # 转型前检查原始 mask，防止 257 等整数转换为 uint8 后伪装成合法值。
    if mask.dtype.kind not in "biu" or not np.isin(mask.values, spec.product.mask.allowed_values).all():
        raise ValueError("Mask must contain only the declared binary integer values")
    values, mean, std, mask = xr.align(values, mean, std, mask, join="exact", copy=False)
    values = values.transpose(*spec.arrays["data"].dimensions).astype(spec.arrays["data"].dtype)
    values = values.assign_coords(channel=values.channel.astype(str),
                                  time=values.time.astype("datetime64[ns]"),
                                  lat=values.lat.astype("float32"), lon=values.lon.astype("float32"))
    mean = mean.assign_coords(channel=values.channel)
    std = std.assign_coords(channel=values.channel)
    mask_coord = spec.product.mask.channel_array_name
    mask = mask.astype(spec.product.mask.dtype).assign_coords(
        {mask_coord: mask[mask_coord].astype(str), "lat": values.lat, "lon": values.lon})
    
    # 仅返回加工后的数组；标准属性与通道元数据由外围 pipeline 统一生成。
    return xr.Dataset({"data": values, "mean": mean, "std": std, "mask": mask})


if __name__ == "__main__":
    from functools import partial

    from pipelines.dataset import build_zarr

    # 修改本次构建的契约、路径及附件；设为 True 时只加工和检查，不写出。
    dry_run = False
    report = build_zarr(
        source_config="/path/to/era5_source.yaml",
        target_spec="/path/to/era5_target.yaml",
        source_path="/data/era5/input.nc",
        destination="/data/output/era5.zarr",
        process=partial(
            process,
            mean_path="/data/stats/era5_mean.nc",
            std_path="/data/stats/era5_std.nc",
            mask_path="/data/stats/era5_mask.nc",
        ),
        dry_run=dry_run,
    )
    print(report.to_dict())
    success = (report.coverage["conformance_passed"] if dry_run
               else report.coverage["written"] and report.quick_passed)
    raise SystemExit(0 if success else 1)
