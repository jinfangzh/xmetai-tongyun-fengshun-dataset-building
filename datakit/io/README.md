# 数据集 IO

`datakit.io` 是数据工作流的接入组件，负责按照声明定位、筛选和组合文件，并管理读取资源。
底层格式解析与写出复用 xarray、Zarr、NumPy 等开源接口；数据集目录、时效和成员等
组织规则由调用方提供。需要处理已有标准文件时，也可以直接使用这些开源接口。

典型使用顺序是：声明数据组织 → 按时间或时效定位文件 → 读取并组合原始数据 →
交给标准化、制作、校验或评测步骤。具体任务入口见
[数据集接入](../../examples/dataset_io/README.md) 和 [工作流](../../pipelines/README.md)。

## 准备工作流输入

具体数据集通过 `config/data_sources/*.yaml` 描述，完整用法见
[统一示例](../../examples/dataset_io/README.md)：

```python
from datakit import io

source = io.open_dataset("/data/fuxi", io.source_config("forecast_fuxi"))
with source.read(selection={"init_time": ["2025-01-01"], "lead_time": [6]}) as raw:
    result = raw.load()
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
| `backends/babj.py` | BABJ 台风报文及源表头保留写回 |
| `config/data_sources/` | 文件组织说明、可选映射及数据集专属入口 |

`io.open(path, layout=...)` 接受布局字典或 YAML 路径。具体源格式由声明和后端决定，
YAML 的 `format.name` 描述数据格式，IO 层选择对应后端，未知格式会报错；
高级调用仍可显式提供 `backend=`。
数据集专属声明位于
`config/data_sources/`；评测角色和报告单位约定位于 `pipelines/evaluation/sources/`。

`schema.DatasetMapping` / `VarSpec` 和 `schema.mapping.apply_dataset_mapping()`
提供通用映射声明与执行能力。`schema.load_mapping()` 接受映射对象、字典或
外部 YAML 路径；数据集默认映射从对应 YAML 的 mapping 声明获取。
IO 不执行变量语义标准化、单位换算、插值或评估配对。

原生 Zarr（缺少 xarray 维度元数据）由上层通过 Zarr 库访问。
Zarr 存储审计工具位于 `datakit.validation.zarr_inspection`。
