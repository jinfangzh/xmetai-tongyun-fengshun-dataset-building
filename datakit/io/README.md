# 数据集 IO

统一入口是 `from datakit import io`，公开接口仅保留 `io.open()`。
IO 描述并访问单个数据集：这个数据集可以是单文件、Zarr Store，或者按路径规则组织的多个文件。
IO 不进行变量语义标准化、单位换算、插值或评估配对。

```python
from datakit import io

source = io.open(
    "/data/forecast", "fuxi_ensemble",
    selection={"init_time": ["2025-01-01"], "lead_time": [6, 12]},
)  # 只保存配置，不扫描或打开数据文件

with source.read() as dataset:
    selected = dataset[["t2m"]].load()
    io.open("/data/output.zarr").write(selected)

with io.open("/data/climatology.nc").read() as dataset:
    print(dataset.data_vars, dataset.coords, dataset.attrs)
```

## 访问操作

- `discover(selection=...)`：只定位并返回文件记录，不打开数据正文。
- `read(selection=...)`：定位并打开匹配文件，返回可关闭的 xarray Dataset。
- `read(load=True)`：加载结果并释放文件句柄，适合原有需要内存数据的调用。
- `iter_datasets(selection=..., chunks=None)`：逐文件访问，推进迭代时关闭前一个文件。
- `write(dataset, ...)`：按目标文件格式或描述布局写出，返回路径列表。

`selection` 使用源坐标名，例如 `time`、`init_time`、`member`、`lead_time`。
读取时的同名条件覆盖构造时条件。精确时间使用 datetime 或 ISO 字符串列表，
路径中的已知字段会先用于定位文件；剩余坐标条件由 xarray 筛选。
slice 条件目前只在打开后筛选，不能缩小目录扫描范围。
描述型集合无条件读取须显式指定 `all_files=True`；单文件和 Zarr 可直接读取。
多文件数组默认 Dask 惰性访问；文本需要立即解析。

默认拒绝覆盖已有文件；需要覆盖时明确指定 `mode="w"`。
Zarr 使用 xarray/Zarr 的默认写入行为；压缩可通过 encoding 显式指定。
NC、GRIB、NumPy、Zarr 的基础操作均由后端使用公开接口实现，没有并行的顶层格式模块。
Zarr 的存储审计工具归属 `datakit.validation.zarr_inspection`，普通读写不依赖这些检查。

## 结构

| 位置 | 职责 |
| --- | --- |
| `dataset.py` | 访问对象、读取条件、文件发现和后端调用 |
| `_description.py` | YAML 校验、可逆路径模板 |
| `descriptions/` | 数据集内部文件组织规则和示例，不声明实例覆盖时间 |
| `backends/files.py` | 路径处理、格式识别、单文件或 Store 读写 |
| `backends/file_collection.py` | 文件发现、路径字段解析、分文件组合和布局写回 |
| `backends/station_tables.py`、`babj.py` | 特殊文本解析 |

文件组织统一使用 descriptions；旧 `readers/` 和顶层格式函数已移除。
格点/站点/多文件气候态的评估 Catalog/Reader 负责请求转换、Schema 映射和
评估契约组装，文件定位与组合复用 IO 后端。
`schema.DatasetMapping` / `VarSpec` 声明映射，
`schema.mapping.apply_dataset_mapping()` 执行惰性转换。
评估在单个文件加载前应用映射，避免先加载全部变量再选择。

## 数据语义描述与评估适配

文件路径组织与可选的 `mapping` 声明放在同一个 `io/descriptions` 文件中。
IO 只解释文件组织，Schema 解析变量与坐标映射，Transform 执行转换。
`schema.load_mapping()` 支持带映射的 IO 描述名称、字典或外部 YAML 路径。
单文件日序气候态的 `data` / `level` 默认约定直接保留在代码中，
不另建描述目录；需要自定义时可传入以下映射：

```yaml
packed_variable: fields
packed_dimension: channel
coordinate_aliases:
  latitude: lat
  longitude: lon
variables:
  z500:
    source: HEIGHT
    scale: 0.10197162129779283
    offset: 0.0
    unit: m
```

日序气候态配置可设置 `mapping: "/path/to/mapping.yaml"`。
简单改名、打包变量拆分和比例/偏移由统一映射执行；365/366 天处理、
环形平滑和插值保留在 pipeline 的气候态取场流程中，仍只读取所需窗口。
原有 `scales`、`units` 配置继续可用，并覆盖映射中相应声明。

`evaluation/adapters` 已移除。统一的 `build_bundle()` 位于
`pipelines/evaluation/execution/sources.py`，将已准备好的 Dataset
封装为 DataBundle，数据角色由调用方指定，不扫描文件或执行数值转换。
数据发现、缓存和按块读取位于 `pipelines/evaluation/execution/inputs`，
使用 IO 或公开的 NumPy/Zarr 接口，再调用 Schema/Transform 与通用适配函数。

原生 Zarr（缺少 xarray 维度元数据）直接通过 Zarr 库访问；
IO 不再提供匿名维度推断或自建 xarray 兼容层。
