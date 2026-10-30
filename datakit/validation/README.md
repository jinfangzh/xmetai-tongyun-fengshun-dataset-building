# 基础校验

本包检查实际数据是否满足声明要求，不依赖 pipelines、运行快照或任务报告。
规则读取 DatasetSchema 或明确的期望值，不从输入数据反推验收标准。

| 文件 | 内容 |
| --- | --- |
| `check_result.py` | CheckResult：检查 ID、状态、类别、期望值、观察值和证据 |
| `structure.py` | 数组布局、坐标、通道、业务属性、附属数组布局；通用天气数组检查 |
| `values.py` | mean/std、二元掩膜、附件值、逐帧质量累积 |
| `comparison.py` | Xarray 标签对齐、分块数值比较、差异统计 |
| `integrity.py` | 内存对象的范围统计、NaN/Inf 计数及时间完整性诊断 |
| `storage.py` | Zarr 版本、编码、分块、块 dtype/shape 与物理 key 检查 |
| `zarr_inspection.py` | 原始元数据读取、摘要、物理块定位及只读块访问 |

基础检查返回 CheckResult 列表。需要为后续步骤提供前置结论或已检查数组时，
返回 `(结果, checks)`；不会直接修改调用方的任务报告。

```python
from datakit.schema import load_schema
from datakit.validation.structure import check_dataset_format
from datakit.validation.values import check_statistics

schema, _ = load_schema("config/dataset_specs/oras5_c88_p25_h24.yaml")
channel_ok, checks = check_dataset_format(dataset, schema)
if channel_ok is not None:
    statistics_ok, statistics_checks = check_statistics(
        dataset["mean"].values, dataset["std"].values,
    )
    checks.extend(statistics_checks)
```

结构检查不读取主数组数值。StandardQualityAccumulator 按完整空间/通道帧更新，
`finish()` 返回 `(checks, quality_metrics)`；扫描顺序、抽样、中断和内存门槛由调用方组织。
单项 pass 不等于产物整体通过，更不代表发布证据完整。

```python
from datakit.validation import compare_datasets

report = compare_datasets(left_dataset, right_dataset, atol=0, rtol=0)
```

比较要求标签集合完整，允许重排，不比较任意属性或存储编码，不做隐式数值变换。
基础维度、坐标、通道和布局检查统一使用 `check_dimensions`、`check_coordinates`、
`check_channels`、`check_same_layout`，返回 CheckResult 列表。五个旧异常式校验接口及其导出已删除。
计算函数、读取和配置错误仍可以抛异常；检查失败后的停止策略由调用方决定。

存储检查与文件访问分开：structure、values、storage 不自行打开文件或遍历目录；
zarr_inspection 提供显式文件访问。生产/转换回执和发布批准状态由
`pipelines.dataset.evidence` 检查，任务结论由 `pipelines.dataset.reports` 汇总。

旧专项的统计量读取、反归一化和 JSON 保存已移至 `pipelines.dataset.old.netcdf_comparison`，
本包不再导出这些文件流程辅助函数。通用偏差、统计汇总和 Dataset 比较继续保留。

通用 Xarray 检查不需要 DatasetSchema：

```python
from datakit.validation import check_dimensions, check_coordinates, check_channels

checks = check_dimensions(array) + check_coordinates(array) + check_channels(array)
failures = [check for check in checks if check.status == "fail"]
```

这些函数读取维度和坐标，不加载主数组，也不因为数据缺陷直接抛异常。
`check_channels(array, coordinate="level")` 可显式检查 level 打包标签。
`check_same_layout(left, right)` 只检查 shape 和维度顺序；标签是否对应需另行比较。
计算函数如 `compute_bias_mean` 会使用布局检查，在前置条件不满足时仍抛 ValueError。
旧 NetCDF 对照入口先汇总基础检查；失败时保存报告、关闭输入并停止后续计算。
