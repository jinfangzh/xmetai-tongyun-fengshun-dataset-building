# 数据集构建与标准校验

本目录独立维护数据集产品定义和标准校验流程。当前先在流程内验证这些定义的适用性，
待共性明确后，再由维护者提炼到公共 `datakit/schema`，以 YAML 声明公共标准。
本次仅调整代码归属、导入和配置位置，保留现有产品规则及 quick/full 判定。

## 目录与职责

```text
pipelines/dataset_building/
├── __init__.py                 # 三个校验流程的入口
├── definitions/                # 流程内部类型、八类产品及显式版本
│   ├── standard_dataset.py     # 产品、网格、掩膜、批次和证据类型
│   ├── _standard_products.py   # 通道语义及辅助结构共用声明
│   ├── dataset_products.py     # 产品汇总及版本选择
│   ├── channel_attributes.py   # 生产通道元数据生成，保留 c116 默认通道
│   └── *_c*.py                # 每类产品的有序通道、网格、掩膜
├── validation/                 # 对已打开对象进行检查、对照和结果建模
│   ├── standard_dataset.py     # 结构、时间、通道、网格、统计量、mask
│   ├── _standard_auxiliary.py  # 已登记附件及权重容差
│   ├── _standard_quality.py    # 主块原值检查及有界统计
│   ├── standard_comparison.py  # 按明确映射比较通道原值
│   ├── evidence.py             # 绑定独立消费者及转换证明
│   └── reports.py              # 结果状态和 quick/full 结论
├── _standard_validation.py     # 单套检查、生产证明及完成结论
├── _standard_scan.py           # 指定块扫描、覆盖计数和中断记录
├── _standard_comparison.py     # 转换及多数据集组合的专项编排
├── _validation_runtime.py      # 当前内存及历史峰值测量
├── config.py                  # JSON 配置校验及类型构建
└── reporting.py               # JSON、CSV、Markdown 报告保存
```

执行示例在 [`examples/dataset_building`](../../examples/dataset_building/README.md)，
配置在其 `configs/` 下，程序回归在 `tests/validation/`。JSON 配置本次保留原格式；
未来公共 schema 的 YAML 设计属于后续任务。

## 依赖与兼容

依赖方向为 `examples → pipelines.dataset_building → datakit`。产品定义、专用检查和
结果类型一起维护，基础库不反向导入流程。严格只读 Zarr IO 仍复用
`datakit/io/_zarr_inspection.py`；通用维度、变量及语义继续保留在公共 schema。
ERA5 c116 的生产常量、网格及来源映射与产品声明统一放在 `definitions/era5_c116.py`，
通道元数据生成放在同层 `channel_attributes.py`。基础库不保留反向导入兼容层。
现有原始数据提取和转换流程改用上述位置，不改变提取、转换或元数据生成规则。

推荐从本包入口调用：

```python
from pipelines.dataset_building import validate_standard_dataset
from pipelines.dataset_building.definitions import DatasetValidationSpec, get_dataset_product
from pipelines.dataset_building.config import load_profile
from pipelines.dataset_building.reporting import write_reports
```

`pipelines.dataset` 原有的三个 `validate_standard_*` 函数继续转发到本包。
产品类型、生产常量、元数据生成、专用检查和报告类型撤出公共基础库，使用方需要更新导入路径：

| 旧位置 | 新位置 |
|---|---|
| `datakit.schema.standard_dataset`、`datakit.schema.dataset_products` | `pipelines.dataset_building.definitions` 下的同名模块 |
| 新增 `datakit.schema.*_c*.PRODUCT` | `pipelines.dataset_building.definitions.*_c*.PRODUCT` |
| `datakit.schema.era5_c116` 及公共 schema 导出的 c116 常量 | `pipelines.dataset_building.definitions.era5_c116` |
| `datakit.schema.channel_attributes` 及公共 schema 导出的元数据生成接口 | `pipelines.dataset_building.definitions.channel_attributes` |
| `datakit.validation.standard_dataset/standard_comparison/evidence/reports` | `pipelines.dataset_building.validation` 下的同名模块 |
| `examples.validation._standard_config/_standard_reports` | 本包的 `config/reporting` |
| `examples.validation.validate_standard_*`、`check_core_consumer` | `examples.dataset_building` 下的同名模块 |
| `examples/validation/profiles` | `examples/dataset_building/configs/profiles` |

完整校验条款与执行方式见 [使用说明](../../docs/DATASET_VALIDATION.md) 和
[校验计划](../../docs/validation/VALIDATION_PLAN.md)。服务器须部署本次代码后才可使用
新的模块路径；本地迁移不会更新服务器，也不会改变历史报告。

`datakit.validation.era5` 的现有布局检查通过参数接收预期通道和网格，没有导入 c116
产品定义，本次保持原位；其中 `validate_era5_c116_layout` 是已有兼容包装。
