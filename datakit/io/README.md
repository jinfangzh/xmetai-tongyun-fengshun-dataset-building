# io

## 1. 定位与边界

`io` 提供本地科学数据格式的显式、原子化读写。它不负责数据集制作、远程存储、
下载和格式自动分发；路径级的组合任务属于 `pipelines`。
`io/readers` 负责具体数据源文件的发现与解码；`schema/semantics.py` 定义
数据类别和时间语义。`evaluation/adapters` 声明源数据到评估契约的映射规则；
`evaluation/execution/sources.py` 按规则组织读取结果并组装
`DataBundle`；`datakit/evaluation/matching.py` 再配对预报与观测。

## 2. 模块与功能

- `netcdf.py`：NetCDF 读取、写入和合并。
- `grib.py`：GRIB 读取、分 edition 读取和写入。
- `numpy.py`：NPY/NPZ 读取和写入。
- `zarr.py`：Zarr 读取、写入、追加、合并和坐标更新。
- `readers/`：特定数据源的文件发现、读取和解码，供评估流程复用。
- `_common.py`：内部路径、变量选择和格式识别辅助函数。

## 3. 公共函数导入

```python
from datakit.io import (
    append_zarr,
    read_grib,
    read_netcdf,
    read_numpy,
    read_zarr,
    write_netcdf,
    write_zarr,
)
```

不存在 `open_data`、`open_dataset`、`write_data` 等统一 dispatch 接口。

## 4. 核心调用示例

```python
from datakit.io import append_zarr, read_netcdf, write_zarr

first = read_netcdf("2025-01.nc")
write_zarr(first, "weather.zarr", chunks={"time": 1})

second = read_netcdf("2025-02.nc")
append_zarr(second, "weather.zarr", append_dim="time")
```
