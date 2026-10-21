# transforms

## 定位

`transforms` 提供可组合的 xarray 数据变换，包括坐标、变量、单位、时间和空间操作。
函数处理内存中的数据对象，不负责打开文件或组织数据集产物。
样本窗口索引属于 `datakit.datasets`，雷达质量控制属于 `datakit.radar`。

## 模块

- `coordinates.py`：维度标准化、坐标规范化、最近点索引。
- `variables.py`：变量名标准化及 channel/level 转换。
- `normalization.py`、`units.py`：归一化、仿射和单位转换。
- `spatial.py`：网格裁剪、经度归位、精确对齐、规则网格插值和网格到站点插值。
- `temporal.py`：累计量转增量、完整小时筛选、重采样、预报步聚合和时间窗累计。

空间或时间插值需要 `.[interpolation]`。

## 示例

```python
from datakit.transforms import crop, normalize_coordinates, resample_time

data = normalize_coordinates(source)
data = crop(data, lat=[30.0, 20.0], lon=[110.0, 120.0])
data = resample_time(data, source_resolution="6min", target_resolution="10min")
```
