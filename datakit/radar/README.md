# radar

## 1. 定位与边界

`radar` 是仅在 core 保留雷达 Dataset 时存在的领域模块，负责雷达源格式解析、显式
网格描述和基础反射率质量控制。批量转换成 Zarr 属于 `pipelines`。

## 2. 模块与功能

- `schema.py`：`RadarGridSpec` 和 `RadarBBox`。
- `cinrad.py`：CINRAD 二进制头解析、文件发现、裁剪和读取。
- `meteonet.py`：MeteoNet NPZ 集合读取。
- `quality.py`：范围、孤立回波、尖峰和静态杂波质量控制。

## 3. 公共函数导入

```python
from datakit.radar import (
    RadarGridSpec,
    RadarQCConfig,
    quality_control_reflectivity,
    read_cinrad,
    read_meteonet_npz,
)
```

## 4. 核心调用示例

```python
from datakit.radar import RadarQCConfig, quality_control_reflectivity

clean = quality_control_reflectivity(
    reflectivity,
    config=RadarQCConfig(min_dbz=0.0, max_dbz=75.0),
)
```
