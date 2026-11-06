# 数据集任务接口

通过 Python API 组织构建、追加和验收任务：

```python
from pipelines.dataset import build_zarr, append_zarr, validate_zarr
```

| 接口 | 职责 | 返回值 |
| --- | --- | --- |
| `build_zarr` | 调用自定义加工、组装标准元数据、检查并写出、回读校验 | `BuildReport` |
| `append_zarr` | 将已有 B 与后续 A 合并到独立 C，完整校验 C | `ValidationReport` |
| `validate_zarr` | 按契约验收已有产物 | `ValidationReport` |

运行与自定义加工示例见 [dataset_building](../../examples/dataset_building/README.md)。

## 模块职责与契约边界

```text
examples：具体任务参数与专属加工
    ↓
pipelines.dataset：编排、资源管理、组装、写出与任务报告
    ↓
datakit：IO、契约模型、基础变换与检查
```

`datakit` 不依赖 `pipelines` 或 `examples`；pipeline 不导入具体示例，也不按数据集名称选择算法。

| 对象 | 负责内容 |
| --- | --- |
| 源契约 | 文件格式、布局与已声明的变量映射 |
| 目标契约 | 通道、网格、时间覆盖、类型、统计量关系、预处理语义及元数据要求 |
| 专属 `process` | 变量组织、派生量、单位换算、重采样、缺测处理、归一化、附件准备和类型转换 |
| 公共 pipeline | 调用加工、管理资源、组装元数据、检查最终产物、写出及回读校验 |

契约声明预期，Python 加工函数执行运算。pipeline 不解释目标契约中的预处理声明来自动执行算法；最终产物检查也不能证明加工步骤实际执行正确。

## 构建

```text
build_zarr(source_config, target_spec, source_path, destination, *, process, read=None, dry_run=False, build_overrides=None, auxiliaries=None)
```

流程为：契约加载和源检查 → 附件路径准备 → 读取源 → `process(raw, source_contract, spec)` → 标准组装 → `create_zarr` → 关闭源 → 复制附件 → `validate_zarr(mode="quick")`。

`auxiliaries` 是契约文件名到源文件路径的映射，例如
`{"const.nc": "/data/static/const.nc", "weight.nc": "/data/static/weight.nc"}`。
正式写出前必须提供全部必需附件；此时只检查名称、文件存在、路径隔离并记录摘要，
不预检查内容。写出后按文件名复制到目标根目录，验证源和副本摘要，不覆盖已有文件，
再由 Quick 检查内容。复制失败保留已写目录并记录 `auxiliary_copy` 失败。
dry-run 不复制，可不提供附件；遗漏的必需项记录在 `coverage.missing_auxiliary_inputs`，
内存符合性通过不代表这些外部附件已验收。附件来源与摘要记录在 `coverage.auxiliary_inputs`。

`build_overrides` 仅支持 `first_time/last_time/dataset_id/content_version/data_revision`，
用于以同一标准构建不同时间段的产物。起止时间须成对提供，覆盖时间时须显式指定
数据集标识；时间步长、通道、网格、类型、质量与附件规则不允许覆盖。
生效的目录名取实际输出目录名，变更身份时不继承旧产物的身份映射证据。
标准 YAML 保持不变，报告 `coverage.target_contract` 记录模板摘要、覆盖参数与
实际生效范围。后续验收此产物也须使用相同运行参数解析后的契约，不能直接按原历史范围验收。

可选 `read(source_path, description, *, spec)` 替换默认读取，返回可关闭的
Dataset，其资源仍由 pipeline 保持到写出结束并关闭。例如 ERA5 小时量与
六小时瞬时量需要保留独立的原始时间轴，由示例提供读取回调；求和和打包
在 `process` 中执行，不在公共 pipeline 中选择专属算法。

源检查覆盖文件发现、可读性、坐标标签唯一性、声明的 packed 结构及已出现映射变量的单位。它不证明未声明的日期覆盖、累计窗口或算法适用性；调用方须选择匹配输入语义和目标要求的加工实现。

`process` 返回包含 `data/mean/std/mask` 的 `xr.Dataset`。数值、dtype、维度顺序、坐标、通道选择和附件由处理函数准备；`mean/std/mask` 等专属参数通过 `functools.partial` 绑定。

`preparation.assemble_standard_dataset` 检查布局和精确对齐，填写根、data 和 channel 的标准属性，不排序、转型或修复数据。主数据可保持惰性；组装在新结果上填写元数据，额外坐标留给最终检查。UTC 等声明仍须由调用方保证真实语义。
准备和组装后的 Dataset 按 `time, channel, lat, lon, mask_channel` 展示维度，
各数组仍保留契约声明的维度顺序。Zarr 没有 Dataset 全局维度顺序，重新读取时
顶部 `Dimensions` 的排列可能改变；主数组的存储维度仍为 `time, channel, lat, lon`。

`process` 可显式复用同模块的 `load_external_array`（加载外部数组并关闭源）、
`prepare_statistics`（加载或计算缺失统计量）、`prepare_standard_arrays`
（精确对齐、mask 转型前检查、目标类型/坐标及 Dask 分块准备）。统计量以
调用方提供的数据域计算；这些工具不执行单位转换或归一化，也不会自动运行。

处理函数自行打开的附件须自行关闭，并保证返回数据在写出时可读取。pipeline 打开的源在检查和写出完成前保持打开，正常及异常退出均释放资源。
加工结果和标准组装结果的关闭回调也在退出读取作用域时执行；正式写出后删除源与产物的
局部引用并执行垃圾回收，再复制附件和进行回读校验。内存预算及校验进程方式不变。
创建检查和写出只调用一次 `create_zarr`，避免重复扫描。

`BuildReport` 提供阶段状态、失败信息、`coverage`、`quick_passed`、`creation_report`、`validation_report` 及 `to_dict()`：

- 正式构建须同时满足 `coverage["written"]` 与 `quick_passed`；quick 通过不等于完整发布验收。
- `dry_run=True` 执行读取、加工及内存检查，不写出、不回读；以 `coverage["conformance_passed"]` 判断结果。
- 已知契约、源检查和组装失败记录到报告；加工异常及意外程序错误继续抛出。
- 拒绝已有输出及输入输出路径重叠。写出或校验失败保留可能存在的目录，不自动删除。

## 构建内部的检查与写出

`create.py` 是 build 的内部实现，不通过包顶层导出独立创建接口。它检查已准备好的 Dataset 的结构、坐标、属性、统计量、掩膜及数值质量，通过后按契约的分块和 Zarr 版本写出；不加工数据或补充业务元数据。

内部检查报告中的 `conformance_passed` 表示输入符合性，`written` 表示写出成功，由 `BuildReport.creation_report` 保留。检查与写出只执行一次；拒绝覆盖，不修改或关闭调用方的 Dataset。对外使用 `build_zarr`，设置 `dry_run=True` 可仅执行加工和内存检查。

## 时间追加

```python
report = append_zarr(
    source="/data/A.zarr", target="/data/B.zarr", destination="/data/C.zarr",
    spec="/path/to/combined.yaml", output="/data/reports/append.yaml",
)
print(report.release_passed)
```

顺序固定为 **B + A → C**。A、B 均须非空，time 为无 NaT、唯一递增的 datetime64，A 首时刻严格晚于 B 末时刻；合并时间轴须精确匹配最终契约。变量与坐标名称、维度、dtype、元数据和静态数组须兼容，通道和网格须符合契约。

A 接受路径或调用方的 Dataset，B 必须是已有 Zarr 目录。路径读取支持 `source_config`、`selection`、`read_options`，固定 `load=False, chunks=None`。C 必须不存在且与输入隔离，不提供覆盖或原地替换选项。

实现复制 B 后逐完整时间片读取 A 并写入 C，保留 B 的压缩、分块和时间编码，按最终契约更新 C 的身份与版本，刷新已有合并元数据。需要完整 BA 的额外磁盘空间，以及至少一个时间片和静态附件的内存；输入在整个操作期间须保持静止。

输入关闭后内部执行 full 校验，按 `release_passed` 判断结果。调用方须提供最终产物的外部附件和发布证据。布局或时间错误在复制前抛出；写出异常向外传播，校验失败返回报告。失败 C 保留，不修改或替换 A、B。

`output` 可保存新的 `.yaml` 校验报告，不得位于 A、B 或 C 内。设计理由、失败处理与取舍见 [ADR-0004](../../docs/adr/ADR-0004-append-to-separate-output.md)。

## 验收

`validate_zarr` 的输入是单份目标产物和契约，检查布局、坐标、类型、存储完整性、统计量、掩膜、附件及已声明的数值与发布规则。它不推断额外的变量物理范围、不执行加工算法，也不读取源数据进行转换对照。加工正确性由独立的源到目标对照能力验证。

```python
report = validate_zarr(
    "/data/product.zarr", "/path/to/target.yaml",
    mode="full", output="/data/reports/validation.yaml",
)
```

公开检查参数为 `mode`、`auxiliary_directory`、`sample_blocks`、`max_blocks`、`memory_budget_mib`、`stop_requested` 和 `on_block`。计数必须是正整数，内存预算须有限且至少为 1 MiB，回调须可调用；非法参数在读取契约和数据前拒绝。
默认校验内存预算为 8192 MiB（8 GiB），构建完成后的 Quick 校验沿用此默认值；
检查仍覆盖预计工作内存及进程当前驻留内存，不预先分配整份预算。

默认 full，按 `release_passed` 判断通过；quick 按 `quick_passed` 判断。快速或中断的扫描不能授权完整发布。数值扫描、物理覆盖和证据完整性分别记录；`on_block`、`max_blocks`、`memory_budget_mib`、`stop_requested` 可控制扫描及中断。

`output=None` 不保存报告。指定时只接受 `.yaml`，其他后缀在读取输入前报错；不能覆盖输入、配置、证据或已有报告。此处的 YAML 约束适用于校验报告，不改变其他接口的文件格式。

报告 identity 中的 `validation_basis` 为 `target_contract_and_declared_rules`，`source_target_comparison` 为 false。契约要求转换回执时，会核验已有回执的摘要和绑定关系；这不代表本次执行了转换对照。缺少必需回执不能通过完整验收，格式错误的证据记录为失败，目标产物检查仍可继续。

固定标准模型和严格 YAML 加载由 `datakit.schema` 提供，具体规范位于 [config/dataset_specs](../../config/dataset_specs)。运行快照和转换回执由 `configuration.load_validation_run` 加载，不写回标准契约：

```python
from pathlib import Path
from pipelines.dataset.configuration import load_validation_run

spec, locations = load_validation_run(Path("/path/to/run.json"))
report = validate_zarr(
    locations["dataset_path"], spec,
    auxiliary_directory=locations["auxiliary_directory"],
)
```

## 文件职责

| 模块 | 职责 |
| --- | --- |
| `build.py` | 契约加载、源检查、加工调用、构建编排及源资源管理 |
| `auxiliaries.py` | 外部附件路径准备、摘要记录及不覆盖的复制；内容交给落盘校验 |
| `preparation.py` | 标准元数据组装；可显式调用的外部数组加载、统计量准备、数组整理、层选择及打包工具 |
| `create.py` | 内存产物检查与写出 |
| `append.py` | B + A 写入独立 C 并完整校验 |
| `validate.py` | 落盘验收、扫描、资源预算及中断 |
| `_sources.py` | 共享读取及资源所有权管理 |
| `reports.py` | 任务报告、检查结果汇总及报告路径保护 |
| `configuration.py` | 运行规格、外部配置、证据引用及摘要验证 |
| `evidence.py` | 生产和转换回执、来源及批准状态检查 |
| `old/` | 格式转换、反归一化导出、NetCDF 拆分、实况提取和专项转换对照 |

`generate_mask` 是自定义加工可主动使用的工具：二值 1 表示陆地、0 表示海洋；比例值按显式阈值生成陆地 mask，海洋取互补。当前 ERA5/ORAS5 示例均读取外部 mask。

`old/` 中的功能仍由包根目录延迟导出，具体调用入口见 [处理示例](../../examples/dataset_processing/README.md) 和 [保留的制作流程](../../examples/dataset_building/history/README.md)。核心构建、追加和验收操作不导入这些实现。
