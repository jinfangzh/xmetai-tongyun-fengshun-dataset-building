# visualization

## 1. 定位与边界

`visualization` 为 NetCDF、GRIB、Zarr 和雷达数据提供轻量查看能力。它只接收已经
读取的数组，不负责文件 I/O 或数据集构建；Matplotlib 和 Pillow 均为可选依赖。

## 2. 模块与功能

- `plots.py`：二维场、等值线、时间序列和垂直廓线。
- `colormaps.py`：雷达、降水、雪和冰雹颜色映射。
- `radar.py`：雷达 RGBA 渲染、PNG 和 GeoJSON 边界叠加。
- `animations.py`：普通场和雷达 GIF 输出。

## 3. 公共函数导入

```python
from datakit.visualization import (
    plot_field,
    plot_profile,
    render_radar_rgba,
    save_field_gif,
    save_radar_png,
)
```

## 4. 核心调用示例

```python
from datakit.io import read_netcdf
from datakit.visualization import plot_field

dataset = read_netcdf("weather.nc")
axis = plot_field(dataset["t2m"].isel(time=0))
axis.figure.savefig("t2m.png", dpi=150)
```
