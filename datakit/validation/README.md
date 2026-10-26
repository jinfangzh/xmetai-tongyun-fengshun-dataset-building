# validation

## 1. 定位与边界

`validation` 对内存中的 xarray 对象执行 schema、时间完整性、无效值和数值范围
检查，并提供跨格式对照、ERA5 日级布局等原有助手。主要接收已打开对象；
`comparison.py` 另有读取 Zarr 统计量和保存诊断 JSON 的配套函数。Zarr 路径级检查由
`pipelines.dataset.check_zarr_range` 组织。

独立 YAML 驱动的构建产物验收是另一套业务：其模型、检查、数值对照、回执、
quick/full 和报告均位于
[`pipelines/dataset_building`](../../pipelines/dataset_building) 的 checks、workflows 和 support。
这些组件没有被其他独立业务广泛复用，故不作为公共基础检查导出。
本模块不反向依赖该流程或可选 core 包，原有接口保持不变。

## 2. 模块与功能

- `schema.py`：核心维度、坐标唯一性和时间顺序检查。
- `integrity.py`：完整年份、NaN/Inf 计数、范围报告和紧凑数据描述。
- `comparison.py`：通用数组结构/坐标/通道对照、反归一化、偏差和统计诊断。
- `era5.py`：ERA5 日级结构、时次和调用方指定的布局检查；保留兼容包装。

已有根导出保持不变；本目录不保留新验收模块的转发副本。

## 3. 公共函数导入

```python
from datakit.validation import (
    analyze_range,
    check_complete_year,
    count_invalid,
    validate_weather_array,
)
```

## 4. 核心调用示例

```python
from datakit.validation import analyze_range, validate_weather_array

validate_weather_array(data)
report = analyze_range(data["data"], axis="level")
print(report.to_text())
```
