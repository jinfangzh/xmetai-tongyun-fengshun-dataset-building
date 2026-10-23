# 数据集 IO

`datakit.io` 提供通用文件读写与多文件组合，不内置数据集名称或专属变量映射。
单文件或 xarray 兼容的 Zarr Store 直接使用 `io.open()`：

```python
from datakit import io

with io.open("/data/input.nc").read() as dataset:
    selected = dataset[["t2m"]].load()
    io.open("/data/output.zarr").write(selected)
```

按数据集组织的访问入口位于 `examples.dataset_io`，例如：

```python
from examples.dataset_io import forecast_fuxi

source = forecast_fuxi.open(
    "/data/forecast",
    ensemble=True,
    selection={"init_time": ["2025-01-01"], "lead_time": [6, 12]},
)  # 只保存配置，不扫描文件

with source.read() as raw:
    # read() 保留源变量名和单位；标准化需要显式调用。
    standard = forecast_fuxi.normalize(raw, ["z500", "t2m"])
    result = standard.load()
```

`io.open_dataset(path, DESCRIPTION)` 接收包含可选变量映射的完整声明，
仅提取原始文件规则，不执行映射。`io.get_description(DESCRIPTION)` 返回完整声明副本，
`io.get_layout(DESCRIPTION)` 返回经过校验的原始 IO 规则。
这些函数接收字典或 YAML 路径；具体数据集由调用方选择，核心库不维护数据集名称注册表。

## 访问操作

- `discover(selection=...)`：只定位并返回文件记录，不打开数据正文。
- `read(selection=...)`：定位并打开匹配文件，返回可关闭的 xarray Dataset。
- `read(load=True)`：加载结果并释放文件句柄。
- `iter_datasets(selection=..., chunks=None)`：逐文件访问，推进迭代时关闭前一个文件。
- `write(dataset, ...)`：按目标文件格式或布局写出，返回路径列表。

`selection` 使用源坐标名，例如 `time`、`init_time`、`member`、`lead_time`。
读取时的同名条件覆盖构造时条件。精确时间使用 datetime 或 ISO 字符串列表，
路径中的已知字段先用于定位文件；剩余条件由 xarray 筛选。
slice 条件目前只在打开后筛选，不能缩小目录扫描范围。
多文件布局无条件读取须显式指定 `all_files=True`；单文件和 Zarr 可直接读取。
多文件数组默认 Dask 惰性访问；单文件使用 xarray 自身的惰性读取，不默认创建 Dask 分块。
文本立即解析。

默认拒绝覆盖已有文件；需要覆盖时明确指定 `mode="w"`。
写回源布局不反向执行变量映射或单位转换，特殊格式可能要求读取时保留的元数据。
Zarr 使用 xarray/Zarr 的默认写入行为；压缩可通过 encoding 指定。

## 结构与边界

| 位置 | 职责 |
| --- | --- |
| `dataset.py` | 访问对象、读取条件和后端调用 |
| `_layout.py` | 显式布局校验、可逆路径模板 |
| `backends/files.py` | 格式识别、单文件或 Store 读写 |
| `backends/file_collection.py` | 文件发现、路径字段解析、组合和布局写回 |
| `backends/station_tables.py` | Diamond 文本和概率表格式读写 |
| `examples/dataset_io/` | 文件组织说明、可选映射及数据集专属入口 |

`io.open(path, layout=...)` 接受布局字典或外部 YAML 路径；不再接受内置数据集名称。
后端选择直接根据文件规则进行：没有布局时使用单文件后端，有布局时使用文件集合后端。
特殊文本由数据集入口通过 `backend=` 显式传入解析模块；通用 IO 不依赖 pipelines。
声明中不再需要 `implementation`，也没有后端名称注册表或动态模块加载。
原 `descriptions/` 已迁入 `examples/dataset_io` 的 Python 数据集模块。
数据集文件开头说明实际组织形式，再声明所需的文件规则和可选映射；接口按数据集差异设置。
数据集文件按 forecast / analysis / reanalysis / observation / climatology 前缀命名；
评估单位口径、GRIB 变量过滤器和评估分组位于评估层。

`schema.DatasetMapping` / `VarSpec` 和 `schema.mapping.apply_dataset_mapping()`
继续提供通用映射声明与执行能力。`schema.load_mapping()` 接受映射对象、字典或
外部 YAML 路径；数据集默认映射从对应的 `examples.dataset_io` 模块获取。
IO 不执行变量语义标准化、单位换算、插值或评估配对。

原生 Zarr（缺少 xarray 维度元数据）由上层通过 Zarr 库访问。
Zarr 存储审计工具位于 `datakit.validation.zarr_inspection`。
