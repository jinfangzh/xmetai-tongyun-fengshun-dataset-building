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

source = io.open("/data/fuxi", io.source_config("forecast_fuxi"))
with source.read(selection={"init_time": ["2025-01-01"], "lead_time": [6]}) as raw:
    result = raw.load()
```

`io.open(path, description=None, selection=...)` 是统一访问入口。
省略 description 时访问单文件或 Zarr；传入字典或 YAML 路径时按描述定位和组合文件。
入口接收包含可选变量映射的完整声明，
仅提取原始文件规则，不执行映射。`io.get_description(DESCRIPTION)` 返回完整声明副本，
`io.get_layout(DESCRIPTION)` 返回经过校验的原始 IO 规则。
这些函数接收字典或 YAML 路径；具体数据集由调用方选择，核心库不维护数据集名称注册表。

## 访问操作

- `discover(selection=...)`：返回公开的 `FileRecord(path, fields)`，不打开数据正文。
- `read(selection=...)`：定位并打开匹配文件，返回可关闭的 xarray Dataset。
- `read(records=...)`：只读取指定记录，不扫描目录或补充文件；缺失文件直接报错。
- `read(load=True)`：加载结果并释放文件句柄。
- `iter_datasets(selection=..., chunks=None)`：逐文件访问，推进迭代时关闭前一个文件。
- `write(dataset, ...)`：按目标文件格式或布局写出，返回路径列表。

`FileRecord` 保存路径与解析字段，不包含评估角色、变量映射或有效时间推导。
`read()` 和 `iter_datasets()` 都可接收已筛选的记录。记录不能为空或重复；
`selection` 只作用于这批文件。无论自动发现还是显式 records，整体空结果和缺失请求标签均报错。
迭代允许单个分区不含某个请求标签，完整消费后检查标签覆盖；
提前结束时调用 `close()` 或使用 `contextlib.closing()`，且不能据此判断完整请求已满足。

```python
records = source.discover(selection={"init_time": ["2025-01-01"], "lead_time": [6]})
with source.read(records=records) as raw:
    result = raw.load()
```

`io.combine_partitions(records, datasets, description=None)` 按位置对应记录和 Dataset，
附加路径坐标、检查结构及重叠记录并组合结果。可传入描述中的结构规则；
不读取文件、不主动加载数据、不关闭输入句柄。调用方应在源关闭前消费或加载组合结果。

`selection` 使用源坐标名，例如 `time`、`init_time`、`member`、`lead_time`。
站点使用 `selection={"station": [54511]}`；文本后端不提供单独的 `stations` 参数。
读取时的同名条件覆盖构造时条件。标量与单元素列表等价，都保留长度为 1 的维度。
标量和列表均按精确标签匹配，时间可用 datetime 或 ISO 字符串；
`"2025-01-01"` 选择当天 00:00，不隐式匹配整天。时间范围显式使用 slice。
非标准日历的字符串按数据索引自身的日历解释，不转成公历。
路径中的已知字段先用于定位文件，剩余条件按坐标筛选，不加载主数据数组来检查标签。
slice 条件目前只在打开后筛选，不能缩小目录扫描范围。
多文件布局无条件读取须显式指定 `all_files=True`；单文件和 Zarr 可直接读取。
多文件数组默认 Dask 惰性访问；单文件使用 xarray 自身的惰性读取，不默认创建 Dask 分块。
文本立即解析。

NetCDF 的打开、数组读取和关闭由 IO 内部的进程内可重入锁保护。
锁可随惰性数组序列化，在每个进程内共享；不在整个 Dataset 生命周期持续持锁。
调用方不应传入 `lock` 或 `backend_kwargs.lock` 覆盖该保护。

所有后端默认拒绝覆盖；需要替换时明确指定 `mode="w"`。
`mode="a"` 和 `mode="r+"` 不属于此接口；标准数据集追加使用 pipeline 的 append 能力。
写出先在目标同目录的临时位置完成，再发布到目标路径。写出失败保留原目标并清理临时产物。
已有 Zarr 目录通过备份、替换和失败恢复实现，不承诺目录交换原子性。
替换需要容纳新产物的额外磁盘空间；仅支持本地、无并发读写的场景。
多文件布局逐文件发布，不承诺整批原子提交，后续文件失败时此前完成的文件不会回滚。

IO 持有的读取上下文尚未关闭时，禁止写入对应源路径、其父目录或子路径，
也检查符号链接和已有硬链接指向的相同源文件。此保护限于当前进程中由 IO 管理的读取；
不追踪调用方自行通过 xarray 打开的句柄。`read(load=True)` 完成加载后释放保护。
写回源布局不反向执行变量映射或单位转换，特殊格式可能要求读取时保留的元数据。
Zarr 使用 xarray/Zarr 的默认写入行为；压缩可通过 encoding 指定。
NPY/NPZ 只写数组值，不保存 xarray 的维度名、坐标和属性；读取时可显式提供 dims 和 coords。
IO 消费 mode、format、compute 等通用写出参数，compute 只允许 True，写出不返回延迟任务。
NumPy 写出不支持其他参数，文本等自定义分支同样拒绝未识别参数。
NetCDF、Zarr、GRIB 的原生参数继续交给后端校验；engine_fallback 仅用于 NetCDF 读取。

## 筛选后按布局写出

```python
with source.read(selection={"init_time": ["2025-01-01"]}) as raw:
    selected = raw[["t2m"]].sel(lead_time=[6], lat=slice(40, 20))
    paths = io.open("/data/subset", io.source_config("forecast_fuxi")).write(selected)
```

读取清单记录源文件的归属；写出时根据当前 Dataset 重新生成计划。
删除变量、筛选时间／成员／时效、裁剪或重排空间坐标后，只写出仍有数据的文件和变量分组。
数组保留当前坐标顺序，并恢复源文件原有的维度结构；IO 附加的路径维度不会写进正文。
源清单没有的记录不会因组合结果中的填充单元而生成新文件。

沿用源布局时，新增或重命名变量、新增坐标标签，以及无法恢复的结构变更会明确报错。
不要删除确定文件归属所需的坐标。需要存储任意加工结果时，可指定固定单文件目标布局，
或直接写入一个 NetCDF／Zarr，使用当前 Dataset 的结构。
Diamond、概率表和 BABJ 支持筛选记录或站点，按源格式恢复行列顺序并保留表头；
必需列和概率阈值集合必须完整，BABJ 表头对应的台风标识不能更改。

全部文件的结构、归属和输出路径检查在开始写出前完成；整体没有可写数据时报错。
编码或底层存储错误仍可能发生在实际写入阶段，多文件发布遵循上述逐文件规则。

## 结构与边界

| 位置 | 职责 |
| --- | --- |
| `dataset.py` | 访问对象、统一筛选、读取上下文和自定义后端接口 |
| `_layout.py` | 显式布局校验、可逆路径模板 |
| `records.py` | 公开文件记录类型 |
| `combination.py` | 分区坐标附加、结构一致性检查与组合 |
| `backends/files.py` | 格式识别、单文件或 Store 读写 |
| `backends/_writing.py` | 源记录筛选投影、本地写出保护、临时产物发布和覆盖恢复 |
| `backends/file_collection.py` | 内置集合访问入口、文件发现、数组分区组合和布局写回 |
| `backends/_text_collection.py` | 文本集合组合、全局记录编号、来源清单和写出规划 |
| `backends/station_tables.py` | 单个 Diamond／概率表的解析、格式恢复和序列化 |
| `backends/babj.py` | 单个 BABJ 报文的解析、表头恢复和序列化 |
| `config/data_sources/` | 文件组织说明、可选映射及数据集专属入口 |

内置访问按“访问接口 → 布局编排 → 单文件格式处理”组织。
格式模块的 `read_file()` 返回单文件 Dataset 和格式元数据；
`prepare_file()` 恢复格式所需的顺序并检查结构，`write_file()` 写入布局层提供的临时路径。
格式模块不发现文件、不分配跨文件编号、不决定输出路径；集合层在所有文件预检查后调度发布。
站点、台风和网格保留各自的组合规则，契约仍使用现有 `format`、`layout` 字段。

`io.open(path, description=...)` 接受描述字典或 YAML 路径。具体源格式由声明和后端决定，
YAML 用 `format: grib` 等标量或 `format: {name: grib}` 映射描述格式，
IO 层统一校验并选择对应后端，未知格式会报错；
高级调用仍可显式提供 `backend=`。
数据集专属声明位于
`config/data_sources/`；评测角色和报告单位约定位于 `pipelines/evaluation/sources/`。

`schema.DatasetMapping` / `VarSpec` 和 `schema.mapping.apply_dataset_mapping()`
提供通用映射声明与执行能力。`schema.load_mapping()` 接受映射对象、字典或
外部 YAML 路径；数据集默认映射从对应 YAML 的 mapping 声明获取。
IO 不执行变量语义标准化、单位换算、插值或评估配对。
评估 profile 的布局适配位于 `pipelines/evaluation/sources/_profile.py`；
评估层按需解码 GRIB 字段，在每个分区中完成映射和换算后再组合，不向 IO 注入加工回调。
evaluation 仅依赖公开 IO 接口。不同气压层等分组若带有冲突的标量坐标，仍明确报错；
本轮不自动丢弃这些坐标或忽略冲突。

原生 Zarr（缺少 xarray 维度元数据）由上层通过 Zarr 库访问。
Zarr 存储审计工具位于 `datakit.validation.zarr_inspection`。
