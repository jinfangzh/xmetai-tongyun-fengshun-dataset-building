# pipelines

## 1. 定位与边界

`pipelines` 是复用 `datakit` 基础能力的流程层，按任务类型建立子包。
`dataset` 承载数据集制作、检查和导出；`evaluation` 承载评测流程、执行策略
和命令行入口。具体数据、日期和输出目录由 `examples/` 的配置提供。

`dataset/` 下划线开头的文件是私有实现，不是额外的公共分类。

数据集接入方式见 [examples/dataset_io/README.md](../examples/dataset_io/README.md)。原始读取与标准化显式分开，
数据集专属规则不再放在 `datakit.io` 的 YAML 中。

## 2. 数据集流程

- 格式转换：本地 NetCDF、GRIB、NumPy、Zarr 之间的目标转换和导出。
- 按时间和通道拆分 NetCDF，提取原始实况。
- 标准数据集校验、转换对比和组合对比。
- Zarr 与 NetCDF 一致性检查。

## 3. 数据集公共函数导入

只从包入口导入，不依赖下划线开头的实现文件：

```python
from pipelines.dataset import (
    convert_to_zarr,
    extract_zarr_to_netcdf,
    validate_standard_dataset,
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
from datakit import io
from datakit.validation import analyze_range

with io.open("dataset.zarr").read() as dataset:
    report = analyze_range(dataset["data"], axis="level")
print(report.to_text())
```

## 5. 评测流程

评测指标、共享数据契约和异常位于 `datakit.evaluation`。
数据源适配、配对、组件注册、集合变换及执行位于 `pipelines.evaluation`：

```text
evaluation/
  sources/       # 数据源适配、变量与评测语义约定、DataBundle 构造
  matching.py    # 预报、实况和参考场配对，生成 EvaluationBatch
  registry.py    # reader、transform、metric、writer 等组件注册
  ensemble.py    # 保留评测契约的集合平均变换
  protocols.py   # 站点降水、台风等场景规则
  execution/     # 切块、缓存、并发执行和归并
  reporting/     # 结果表、派生视图和报告
```

`sources/` 调用 `datakit.io` 和 `examples.dataset_io` 的原始读取接口，
原始数据集模块不依赖评测流程。`serial.py` 保留专用串行入口。
Fengqing 应用配置位于
`examples/evaluation/configs/`，在仓库根目录可按配置名运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
```
