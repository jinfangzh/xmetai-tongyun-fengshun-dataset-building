# 基础数据对照示例

本目录保留 `validate_netcdf.py`、`validate_zarr_nc.py` 等现有 NetCDF/Zarr 对照示例。
它们用于格式转换前后的基础比较，保持原有用途。

标准数据集的 quick/full 验收和转换数值对照入口现位于
[`examples/dataset_building`](../dataset_building/README.md)，配置集中在
`examples/dataset_building/configs/`。对应实现位于
[`pipelines/dataset_building`](../../pipelines/dataset_building/README.md)。
组合检查、core 探测及消费者/模型回执核验已移除，不属于当前数据构建验收范围。

完整运行、测试及报告说明见 [标准数据集校验](../../docs/DATASET_VALIDATION.md)。
