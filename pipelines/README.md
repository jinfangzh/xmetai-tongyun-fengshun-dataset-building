# pipelines

## 1. 定位与边界

`pipelines` 是复用 `datakit` 基础能力的流程层，按任务类型建立子包。
`dataset` 承载数据集制作、检查和导出；`evaluation` 承载评测流程、执行策略
和命令行入口。具体数据、日期和输出目录由 `examples/` 的配置提供。

`datakit.schema.datasets` 提供固定产品标准模型、严格 YAML 加载和独立标准文件；
`dataset_building` 负责严格校验、扫描、执行配置加载和报告保存；
配置与 CLI 位于 `examples/dataset_building/`。详见
[数据集构建与标准校验](dataset_building/README.md)。其下 production、checks、workflows、support
直接并列，不再嵌套 acceptance。生产来源映射和元数据保持独立；checks 区分格式、数值、
存储和转换规则；workflows 组织 quick/full、批量执行和转换对照；
support 提供运行/转换对照规格、执行输入及报告。转换回执在转换规则中核验。现有 `dataset` 中的单套验收、
转换对照两个 `validate_standard_*` 函数保留转发，基础库不依赖此流程包。

`dataset/_conversion.py`、`dataset/_datasets.py`、`dataset/_statistics.py`、
`dataset/_radar.py` 和 `dataset/_ensemble.py` 只是控制单文件体积的私有实现，
不是额外的公共分类。

## 2. 数据集流程

- 格式转换：本地 NetCDF、GRIB、NumPy、Zarr 之间的目标转换和导出。
- 数据集检查与按年份分区。
- 统计产物：归一化、差分、气候态和分位数文件构建。
- 雷达数据集：CINRAD 和 MeteoNet 转 Zarr。
- 集合数据集：按日期和预报时效组织 NetCDF 成员并写入 Zarr。
- 路径检查：打开 Zarr 并生成逐通道或逐层范围报告。

## 3. 数据集公共函数导入

只从包入口导入，不依赖下划线开头的实现文件：

```python
from pipelines.dataset import (
    build_normalization_statistics,
    check_zarr_range,
    convert_cinrad_to_zarr,
    convert_ensemble_netcdf_to_zarr,
    convert_to_zarr,
)
```

## 4. 数据集调用示例

```python
from pipelines.dataset import convert_to_zarr

convert_to_zarr(
    source="input.nc",
    destination="dataset.zarr",
    chunks={"time": 1, "lat": 256, "lon": 256},
)
```

```python
from pipelines.dataset import check_zarr_range

report = check_zarr_range("dataset.zarr", variable="data", axis="level")
print(report.to_text())
```

## 5. 评测流程

评测数据契约、配对和指标位于 `datakit.evaluation`；
`pipelines.evaluation` 将它们接成完整流程，`reporting/` 负责结果表、
派生视图和报告文件。Fengqing 应用配置位于
`examples/evaluation/configs/`，在仓库根目录可按配置名运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
```
