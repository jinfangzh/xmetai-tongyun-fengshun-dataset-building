# 基础数据对照示例

本目录保留 `validate_netcdf.py`、`validate_zarr_nc.py` 等现有 NetCDF/Zarr 对照示例。
它们用于格式转换前后的基础比较，保持原有用途。

标准数据集的 quick/full、转换、组合和 core 专项入口现位于
[`examples/dataset_building`](../dataset_building/README.md)，配置集中在
`examples/dataset_building/configs/`。对应实现位于
[`pipelines/dataset_building`](../../pipelines/dataset_building/README.md)。

完整运行、测试及报告说明见 [标准数据集校验](../../docs/DATASET_VALIDATION.md)。
