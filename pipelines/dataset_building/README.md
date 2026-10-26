# 数据集构建：生产定义与产物验收

本目录按职责组织生产定义、检查规则、执行流程和验收支撑，不再嵌套 acceptance 目录。
生产定义保持独立，检查规则不以生产默认值或生成结果作为验收真值。
检查、扫描、回执和报告服务于本套工作流；固定标准模型及 YAML 解析由基础库提供。
十套验收标准各自独立保存在 [`datakit/schema/datasets`](../../datakit/schema/datasets)，
不再同时维护 JSON profile 和 Python 产品注册表。已有 quick/full 扫描与证据门槛保留。

## 目录与职责

```text
pipelines/dataset_building/
├── __init__.py                 # 四个公共入口；延迟加载对应工作流
├── production/                 # 原有生产能力，不是验收标准
│   ├── __init__.py
│   ├── channel_metadata.py     # 生产通道元数据生成，保留 c116 默认通道
│   └── era5_c116.py            # c116 生产通道、来源映射和目标坐标
├── checks/                     # 检查规则；不打开文件、不组织扫描
│   ├── __init__.py
│   ├── format.py              # 主数据/附件布局、坐标、时间及语义
│   ├── values.py              # 统计量、掩膜、附件值和主数据质量累计
│   ├── storage.py             # 编码、物理覆盖、冻结证明及完整性规则
│   └── conversion.py          # 源—目标对应、误差及转换回执规则
├── workflows/                  # 读取、扫描、资源、中断及批量执行
│   ├── __init__.py
│   ├── dataset.py             # 单套与批量验收，共用 quick/full 扫描
│   └── conversion.py          # 转换对照与配置驱动执行；不是转换器
└── support/                    # 验收支撑；不反向导入规则、流程或生产定义
    ├── __init__.py
    ├── specifications.py      # 运行快照/回执规格及源目标转换对照配置
    ├── inputs.py              # 调用基础库加载标准，装配执行 JSON 及回执
    └── reports.py             # 结果模型、结论汇总、输出保护及保存
```

执行示例在 [`examples/dataset_building`](../../examples/dataset_building/README.md)，
专项执行配置在其 `configs/` 下，程序回归在 `tests/dataset_building/acceptance/`。
单套 YAML 是验收声明的唯一维护来源，合并草稿不参与运行。
附件不再单独成模块：布局与角色进入 `checks/format.py`，数值与掩膜关系进入
`checks/values.py`。布局通过后才允许相应的数值读取和检查。
quick/full 共用 `workflows/dataset.py` 的扫描和资源辅助，不拆出独立扫描包。
原运行器按数据集、转换两条流程并入对应工作流，转换配置装配统一在 `support/inputs.py`。
转换回执业务核验在 `checks/conversion.py`，不另建 evidence 模块；文件读取、摘要检查由流程组织。
`support/reports.py` 仅负责结果、状态汇总和输出；JSON/CSV/Markdown、退出码和验收门槛不变。

## 依赖与兼容

依赖方向为 `examples → workflows → checks → support → datakit`，流程和规则也可直接使用基础标准及 IO。
`support` 不反向导入规则和流程；验收不导入 `production`，基础库不反向导入本包。
`datakit.schema.datasets.specifications.DatasetSchema` 保存固定目标规则，模型与约束不读取被检数据。
`support.specifications.DatasetValidationSpec`
另加运行快照、外部回执及执行配置摘要，两者不混写。
`datakit.schema.datasets.load_schema` 读取独立 YAML，`support.inputs.load_validation_schema` 在此基础上构造
尚未绑定回执的运行实例，`load_validation_run` 再装配独立执行 JSON。
两个原同名加载函数现在显式区分，返回类型及校验语义保持不变。
同一套验收内共享的检查、证据和报告模型集中在本包，基础库无反向兼容层。

生产定义只依赖基础库，不依赖验收包；验收也不导入生产定义生成期望值。
包入口延迟导入验收功能，因此单独导入生产模块不会加载验收程序。
已有生产模块迁入 `production/` 时保持内容不变，原始数据提取和转换仅更新导入路径。
严格只读 IO 仍复用 `datakit/io/_zarr_inspection.py`；原有通用检查、转换和统计保持原位。

推荐从本包入口调用：

```python
from pipelines.dataset_building import validate_standard_dataset
from datakit.schema.datasets import load_schema

spec, locations = load_schema("datakit/schema/datasets/era5_land_c8_1p5_h24.yaml")
report = validate_standard_dataset(locations["dataset_path"], spec, mode="quick")
```

需要保存批次报告时调用 `run_dataset_validation(output, schemas=[...])`；
有冻结生产证明时改用 `run_configs=[...]`。`workflows/dataset.py` 处理加载、扫描、逐块凭据和落盘，
示例只处理 CLI 参数、信号和退出码。本流程只检查构建产物，不调用外部模型读取器、
训练或推理，也不接收消费者及模型测试回执。

`pipelines.dataset` 的 `validate_standard_dataset`、`validate_standard_conversion`
继续转发到本包；组合验收入口已删除。
验收代码的最终位置如下；旧内部路径不建立反向兼容层：

| 旧位置 | 新位置 |
|---|---|
| 固定标准模型（原 `support.specifications` 的标准部分） | `datakit.schema.datasets.specifications` |
| YAML 解析及标准源完整性（原 `support.inputs` 的标准部分） | `datakit.schema.datasets.inputs` |
| 运行快照、回执及转换对照规格 | `support.specifications` |
| 执行 JSON 装配、回执绑定和执行源完整性 | `support.inputs` |
| Python 产品 `PRODUCT` 和 `dataset_products.py` | 单套 YAML，不恢复 Python 产品注册表 |
| 原 `production.channel_attributes` | `production.channel_metadata`；`production.era5_c116` 不变 |
| 原 `acceptance.checks.structure/auxiliary/quality` | 按格式、数值拆入 `checks.format`、`checks.values` |
| 原数据集流程中的存储和冻结证明规则 | `checks.storage`；文件遍历仍由流程负责 |
| 原 `acceptance.dataset_validation`、数据集运行器 | `workflows.dataset` |
| 原 `acceptance.conversion`、转换运行器 | 规则进入 `checks.conversion`，执行进入 `workflows.conversion` |
| 原独立 evidence、报告中的转换回执核验 | `checks.conversion.attach_conversion_evidence`，不保留独立 evidence 模块 |
| 原报告、reporting | `support.reports` |
| `examples.validation.validate_standard_dataset/validate_standard_conversion` | `examples.dataset_building` 下的同名模块 |
| 旧 JSON profiles 与 inventory | `datakit/schema/datasets/*.yaml`；CLI 使用 `--schema` 或 `--schema-directory` |

YAML 只维护目标标准；执行 JSON 通过 `schema` 和嵌套 `execution` 绑定
`snapshot_id`、production/conversion evidence 及其 SHA256。
转换专项 JSON 的 `target_schema` 也使用同样的 `execution` 结构。
执行参数不能覆盖规则或关闭 YAML 已要求的转换检查。JSON 解析结果与其摘要来自
同一份字节，运行前后检查源文件未变化。报告输出不得覆盖输入数据、配置或证据。

完整校验条款与执行方式见 [使用说明](../../docs/DATASET_VALIDATION.md) 和
[校验计划](../../docs/validation/VALIDATION_PLAN.md)。服务器须部署本次代码后才可使用
新的模块路径；本地迁移不会更新服务器，也不会改变历史报告。

`datakit.validation.era5` 的现有布局检查通过参数接收预期通道和网格，没有导入 c116
产品定义，本次保持原位；其中 `validate_era5_c116_layout` 是已有兼容包装。

### 执行顺序

1. `datakit.schema.datasets.load_schema` 读取、审查标准，生成目标模型；不从待检数据反推期望值。
2. `support.inputs.load_validation_schema`、`load_validation_run` 准备运行规格、绑定路径和回执；
   不修改标准，固定标准结构和推导方法由 `datakit/schema/datasets/specifications.py` 提供，
   运行结构由 `support/specifications.py` 提供。
3. `workflows/dataset.py`、`conversion.py` 读取数据和回执，按前置条件调用 `checks` 中的规则。
4. 工作流记录覆盖和中断状态，使用 `support/reports.py` 汇总及保存报告。

保留两个数据验收 CLI：`validate_standard_dataset.py`、`validate_standard_conversion.py`，
及对应的校验函数和运行器。单套入口仍可批量独立验收多份 YAML，不要求各套时间或网格相同。
工具版本 `1.8` 删除组合 CLI、配置模板、校验函数和报告中的
`cross_dataset_compatibility`、`combination.scope`；不再输出联合使用结论。
此前工具版本 `1.7` 已删除消费者探测、模型回执核验及报告中的消费者兼容栏目。
旧配置中的 `consumer_required`、`consumer_evidence`、`model_evidence` 会明确报错，
即使要求值为 false 也不再接受。数据生产与转换回执仍保留，历史报告不改写。
内部导入路径按上表迁移，不在旧位置复制实现；quick 不变成 full，缺生产或转换证据不放行。
