# 数据集操作与验收

公共入口保持从 `pipelines.dataset` 导入。基础检查集中在
[`datakit.validation`](../../datakit/validation/README.md)，本目录组织读取、创建、
扫描、证据检查和任务报告，不维护另一套数组检查实现。

```python
from pipelines.dataset import create_zarr, append_zarr, validate_zarr, compare_zarr
```

| 函数 | 职责与返回值 |
| --- | --- |
| `create_zarr(dataset, destination, spec, ...)` | 检查准备好的 Xarray Dataset，通过后写出，返回 ValidationReport |
| `append_zarr(source, target, ...)` | 固定沿时间追加完整帧，返回目标 Path |
| `validate_zarr(target, spec, output=None, ...)` | 验收已有产物，返回 ValidationReport |
| `compare_zarr(source, target, ...)` | 对齐完整标签集合后比较数值，返回差异字典 |

## 创建

```python
report = create_zarr(
    dataset, "/data/new/product.zarr", "config/dataset_specs/oras5_c88_p25_h24.yaml",
    dry_run=True, output="/data/reports/create.json",
)
```

输入必须是外围准备好的 `xr.Dataset`；spec 接受 DatasetSchema 或 YAML 路径。
创建不会重排通道、转换 dtype、归一化、合并附属变量或补充业务属性。
数组结构、坐标、通道、属性、统计量、掩膜和数值质量通过检查后，才按 spec
的分块和 Zarr 版本写出。继承的存储编码会在副本中清除，输入对象不会被关闭或修改。

`dry_run=True` 只检查；`coverage.conformance_passed` 表示输入符合性，
`coverage.written` 表示写出成功。始终拒绝覆盖目标；不符合时返回失败报告且不创建目标。
创建报告不代表完整发布验收：外部附件、物理存储与发布证据需要 `validate_zarr` 检查。

## 时间追加

```python
append_zarr(next_days, "/data/product.zarr")
```

固定沿 `time` 追加。源和目标要求相同变量、兼容 dtype、相同通道及非时间坐标、
相同静态数组。时间坐标必须为无 NaT、唯一且递增的 datetime64，新时间严格晚于已有时间。
不检查时间频率或连续性，不自动更新覆盖范围、版本或冻结证据。

追加与比较支持内存 Dataset 或源路径；追加路径读取可提供 `source_config`、
`read_options` 和 `selection`。调用方的 Dataset 保持打开；函数自行打开的输入会关闭。
追加沿用目标编码并刷新已有合并元数据；写入不是原子事务，需要串行管理写入。

## 产物验收

```python
report = validate_zarr(
    "/data/product.zarr", "config/dataset_specs/oras5_c88_p25_h24.yaml",
    output="/data/reports/validation.json",
)
```

默认 `mode="full"`；抽样使用 `mode="quick"`。可传 `on_block`、`max_blocks`、
`memory_budget_mib`、`stop_requested` 控制扫描与中断。
快速或中断的扫描不能授权完整发布。数值扫描、物理覆盖与证据完整性分别记录。

`output=None` 不保存文件；指定时写单份 JSON，不覆盖输入、配置、证据或已有报告。
若需生产快照和转换回执，在外围加载运行配置：

```python
from pathlib import Path
from pipelines.dataset.configuration import load_validation_run

spec, locations = load_validation_run(Path("config/validation/run.json"))
report = validate_zarr(
    locations["dataset_path"], spec,
    auxiliary_directory=locations["auxiliary_directory"],
)
```

固定标准模型及严格 YAML 加载由 `datakit.schema` 提供，具体规范位于
[`config/dataset_specs`](../../config/dataset_specs)。运行信息不写回标准 YAML。

## 比较

```python
difference = compare_zarr(source, target, mapping=None, atol=0, rtol=0)
```

允许坐标顺序和变量维度顺序不同，但要求数组、维度和坐标标签集合完整对应，
不静默取交集。可通过 mapping 字典或 YAML 显式指定源到目标的
`dimensions`、`variables`、`channels` 名称映射。

不改变单位或数值，不比较任意属性、存储编码和分块。对应 NaN 视为相等，
默认精确比较，可配置容差。`memory_budget_mib` 限制逻辑块，不能约束后端解码缓冲区。
数值比较直接复用 `datakit.validation.comparison.compare_datasets`。

## 文件职责

```text
pipelines/dataset/
├── create.py                  # 输入符合性检查与写出
├── append.py                  # 时间追加
├── validate.py                # 验收、扫描、资源及中断
├── compare.py                 # 输入读取、名称映射与比较调用
├── _sources.py                # 共享读取与资源所有权
├── reports.py                 # 任务报告、发布结论、输出路径保护
├── configuration.py           # DatasetValidationSpec、运行 JSON、证据引用及摘要验证
├── evidence.py                # 生产/转换回执、来源及批准状态检查
└── old/                       # 保留的旧处理逻辑
    ├── conversion.py          # 旧格式构造与反归一化导出
    ├── channel_metadata.py    # 旧业务通道属性生成
    ├── netcdf_comparison.py   # 旧 Zarr/NetCDF 专项对照
    ├── conversion_validation.py # 转换规则、配置加载和专项验收
    ├── raw_truth.py           # ERA5 单日参考数据提取及专用定义
    └── split.py               # NetCDF 时间/通道拆分
```

原 checks、workflows、support 目录和 zarr_dataset.py 已移除，不保留内部路径转接。
对外通过包根目录调用公共函数；旧公共函数延迟指向 old 中的实现。`extract_raw_truth_day`、`convert_to_zarr`、
NetCDF 导出/拆分及转换专项验收仍保留；它们没有被四个核心函数完全替代。

四个操作直接使用本页介绍的 Python API。
ERA5/ORAS5 制作及转换验收入口见 [`examples/dataset_building/history`](../../examples/dataset_building/history/README.md)。
更详细的证据和发布要求见 [验收说明](../../docs/DATASET_VALIDATION.md)。

旧转换专用的 `ConversionChannelRule`、`ConversionValidationSpec`、`load_conversion_run`
也集中在 `old/conversion_validation.py`，当前 configuration 不再包含它们。
四个核心操作不导入 old。当前验收使用的转换回执核验仍留在 evidence.py。

根目录只保留四个核心操作及其直接/间接依赖。ERA5 提取和 NetCDF 拆分也归入 old，
这里的 old 表示核心操作以外的保留功能，不表示已禁用；包根目录公共调用方式不变。

运行规格 `DatasetValidationSpec` 与配置加载统一在 configuration.py；pipeline 的 specifications.py 已删除。基础标准模型仍在 datakit.schema。

old/netcdf_comparison.py 已改用结果式基础检查：输入结构不符合时不读取统计量或执行反归一化，
失败报告保留检查 ID、状态和原因，并继续以 ValueError 向调用方报告失败。
其统计量读取、反归一化及 JSON 保存均在该旧流程内维护，不再从 datakit.validation 导出。
