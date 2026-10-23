# validation

## 1. 定位与边界

`validation` 对内存中的 xarray 对象执行 schema、时间完整性、无效值和数值范围
检查。使用 `datakit.io` 打开数据后，可直接调用 `analyze_range()`；
标准数据集的路径级校验由 `pipelines.dataset.validate_standard_dataset` 组织。

## 2. 模块与功能

- `schema.py`：核心维度、坐标唯一性和时间顺序检查。
- `integrity.py`：完整年份、NaN/Inf 计数、范围报告和紧凑数据描述。

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
