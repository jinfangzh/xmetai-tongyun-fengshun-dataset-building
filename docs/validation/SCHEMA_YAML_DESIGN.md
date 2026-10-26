# 单套标准数据集 YAML 设计说明

本说明定义单套数据集 YAML 的通用结构、字段职责和关联方式，适用于 CRA、ERA5、ERA5-Land、ORAS5、S2S 等现有产品。各文件使用相同的组织方式，具体通道、时间、网格和处理规则按产品实际标准填写。

## 目录

1. [用途与文件组织](#1-用途与文件组织)
2. [通用结构目录](#2-通用结构目录)
3. [第一部分：meta_info](#3-第一部分meta_info)
4. [第二部分：variable_metadata](#4-第二部分variable_metadata)
5. [两部分的关联规则](#5-两部分的关联规则)
6. [各产品的适用差异](#6-各产品的适用差异)
7. [编写与维护约定](#7-编写与维护约定)
8. [与合并文件的对应关系](#8-与合并文件的对应关系)
9. [程序接入与验证边界](#9-程序接入与验证边界)

## 1. 用途与文件组织

每份独立 YAML 描述一套数据的身份、存储结构、坐标、通道、变量语义及校验要求。它既不是 Zarr 元数据的直接导出，也不是数据验收报告：Zarr 记录数据实际保存的声明，YAML 提供独立的预期标准。

配置与设计文档的文件组织如下：

```text
datakit/schema/datasets/
├── cra_c71_p25_h6.yaml
├── cra_c87_p25_h24.yaml
├── cra_c87_1p5_h24.yaml
├── era5_c78_p25_h6.yaml
├── era5_c87_p25_h24.yaml
├── era5_c116_p25_h6.yaml
├── era5_land_c8_1p5_h24.yaml
├── oras5_c88_1p5_h24.yaml
├── oras5_c88_p25_h24.yaml
└── s2s_c76_1p5_h24.yaml

docs/validation/
├── SCHEMA_YAML_DESIGN.md
└── SCHEMA_YAML_MERGE_DESIGN.md
```

独立文件以 [CRA c71](../../datakit/schema/datasets/cra_c71_p25_h6.yaml) 为布局示例。
十份独立文件是唯一运行时标准。原草稿目录仅保留未采用的 `datasets.yaml`，
与 [合并设计方案](SCHEMA_YAML_MERGE_DESIGN.md) 一起作为历史设计记录，不参与加载或同步维护。

## 2. 通用结构目录

以下是字段的逻辑层级，不是磁盘目录。`<变量名>`、`<文件名>` 表示按本产品实际内容填写的键；共享层次和特殊参数只在适用时声明。

```text
meta_info
├── name
├── dataset_id
├── expected_root_dataset_id
├── product
│   ├── id
│   ├── version
│   └── variant
├── template_version
├── schema_version
├── status
├── dataset_path
├── auxiliary_directory
├── dataset_version
├── storage_format
│   ├── name
│   ├── version
│   └── allow_nested_groups
├── data_revision
├── validation_scope
├── time_coverage
│   ├── first
│   ├── last
│   ├── time_count
│   ├── time_resolution
│   │   ├── value
│   │   └── unit
│   └── timezone
├── grid
│   ├── space_resolution
│   ├── coordinate_units
│   ├── latitude
│   │   ├── first
│   │   ├── last
│   │   ├── count
│   │   └── direction
│   └── longitude
│       ├── first
│       ├── last
│       ├── count
│       └── direction
├── pressure                         适用时声明
│   ├── variables
│   ├── units
│   ├── channel_name_template
│   └── levels
├── depth_below_sea_surface          适用时声明
│   ├── variables
│   ├── units
│   ├── channel_name_template
│   └── level_codes
├── channels
├── arrays
│   ├── data
│   │   ├── dimensions
│   │   ├── dtype
│   │   └── chunks
│   ├── time
│   ├── channel
│   ├── lat
│   ├── lon
│   ├── mean
│   ├── std
│   ├── mask
│   └── mask_channel
├── mask
│   ├── channels
│   ├── allowed_values
│   ├── valid_value
│   └── complement_sum
├── metadata
│   ├── data_attributes
│   │   ├── data_representation
│   │   ├── normalization_method
│   │   ├── mean_variable
│   │   ├── std_variable
│   │   ├── scale_factor_location
│   │   ├── add_offset_location
│   │   └── inverse_transform
│   └── channel_metadata
│       ├── attribute
│       ├── version_attribute
│       ├── version
│       ├── require_exact_channels
│       ├── long_name_required
│       ├── scale_factor_from
│       ├── add_offset_from
│       └── scale_offset_formula
├── normalization
│   ├── statistics
│   │   ├── mean_finite
│   │   ├── std_finite
│   │   └── std_positive
│   └── effective_statistics         适用时声明
│       └── <变量名>
│           ├── mean
│           └── std
└── auxiliaries
    ├── check_variable_names
    ├── unregistered_file_status
    └── files
        └── <文件名>
            ├── format
            ├── role
            ├── required
            ├── dimensions
            ├── dtype
            ├── grid
            ├── finite
            ├── channels             通道型附件适用
            ├── nonnegative_atol     权重适用
            └── require_positive_value

variable_metadata
└── <变量名>
    ├── units
    ├── long_name
    ├── level_type                   单层变量适用
    ├── level                        固定高度或土壤层索引适用
    ├── level_units                  有level时声明
    ├── source_name
    ├── source_units
    ├── variable_type
    ├── preprocessing
    └── unconfirmed_fields           存在明确未确认字段时声明
```

`pressure` 与 `depth_below_sea_surface` 是当前产品使用的两类共享层次，并非所有文件都必须填写的公共模板。ERA5-Land 没有共享多层表，直接在变量条目中声明各土壤层索引。

`arrays` 中每个数组均声明 `dimensions` 和 `dtype`，结构目录仅对 `data` 展开这两个公共字段。附件的 `nonnegative_atol` 与 `require_positive_value` 仅用于权重。

## 3. 第一部分：meta_info

### 3.1 标识、位置与版本

| 字段 | 含义与约束 |
| --- | --- |
| `name` | 供人阅读的数据集名称，可包含产品、分辨率和频次 |
| `dataset_id` | 产品与变体的短配置标识，例如 `cra.c71.p25.h6`；不等同于现有 Zarr 根属性中的历史长标识 |
| `expected_root_dataset_id` | 要求 Zarr 根属性保存的长标识；与目录名不同须填写 `identity_mapping_evidence` |
| `product` | 明确的 `id/version/variant`，不从文件名或通道数量推断产品 |
| `template_version` | 本套模板规则的版本，当前为 `v1`；与 `dataset_id` 共同标识所采用的目标标准，覆盖第一部分规则及第二部分变量定义 |
| `schema_version` | 要求根属性中的 schema 版本；当前为字符串 `2.0`，不是 ORAS5 产品版本 |
| `status` | `approved` 或 `draft`；草稿不具备已批准标准的放行资格 |
| `dataset_path` | 本套主数据的实际路径，不从配置标识自动推断 |
| `auxiliary_directory` | 登记附件的查找目录，可以与主数据目录相同 |
| `dataset_version` | 数据内容版本标签，不是 YAML 格式版本或软件包版本 |
| `storage_format` | 存储格式要求；`version: 3` 表示 Zarr 存储格式 v3，不表示 Python zarr 库版本 |
| `data_revision` | 本套数据的修订标识，不把它自动解释为时间覆盖的结束日期 |
| `validation_scope` | 校验范围标识；当前为 `confirmed-format`，不是数据已通过的状态 |

`storage_format.allow_nested_groups: false` 要求不出现嵌套组。允许的根数组由 `arrays` 清单另行确定，两类要求不能互相替代。

`template_version` 各套独立维护。它不表示数据内容版本，也不要求 Zarr 已经保存同名属性。
验收程序显式加载所选文件，并在报告中记录配置标识、模板版本及源文件 SHA256。
相对数据、附件和证据路径相对 YAML 所在目录解析。

通道及顺序、单位、层次、预处理、数组格式或校验要求发生实质变化时，升级模板版本，例如 `v1` → `v2`。仅修改注释、排版、路径，或在原频次与规则下追加时间数据，通常不升级模板版本；相应实例信息仍单独更新。即使版本不变，字节修改也会改变 SHA256，须重新运行并绑定报告/回执。发布新版本时应保留旧版本的不可变文件或 Git 提交引用，不能用同一标识和版本覆盖不同规则。版本字段本身不保存历史内容。

### 3.2 时间与水平网格

`time_coverage` 描述本套数据声明的时间覆盖：`first`、`last` 为首末时次，`time_count` 为时间点数量，`timezone` 为时区。

时间间隔采用以下形式：

```yaml
time_resolution: {value: 6, unit: hour}
```

当前十套数据均以 `hour` 为单位。间隔的数量和单位必须同时明确；扩展其他单位时，需要先明确相应换算与校验规则。`datetime64[ns]` 中的 `ns` 是时间值精度，与六小时、二十四小时等采样间隔不同。

对固定间隔时间轴，首末时次、间隔和数量必须自洽；校验应独立生成完整时间轴并比较实际值，不能只比较端点。

`grid` 声明水平间隔和坐标单位。`latitude`、`longitude` 分别保存坐标首值、末值、数量及方向。每个坐标轴的端点、长度、方向和间隔应相互一致，并与实际完整坐标比较。

### 3.3 共享层次与完整通道

| 字段 | 描述内容 | 关联方式 |
| --- | --- | --- |
| `pressure` | 适用变量、气压单位、层表及通道名模板 | 例如变量 `q` 与合法层号 `850` 按声明模板对应 `q850` |
| `depth_below_sea_surface` | 海洋变量、深度单位、通道名模板及显式标签映射 | ORAS5 的 `19978` 标签对应 `199.78 m`，不按任意数字后缀推断 |
| `channels` | 本套完整通道名称及唯一顺序 | 必须逐项检查，不能用层表或变量字典的展开顺序替代 |

共享层次只解释通道与物理层值的关系，不要求 Zarr 新增同名数组或维度。气压层和海洋深度定义分别适用于各自产品，不能复制 CRA 层表作为其他产品标准。

### 3.4 数组布局与内嵌掩膜

`arrays` 描述各数组的名称、维度顺序和原始逻辑类型。主数据 `chunks` 中的 `full` 表示对应维度的完整长度；它是本配置的表达约定，不是原生 Zarr 分块字段值。

数组形状由已有声明共同确定，不重复填写整套 shape：

| 维度 | 长度来源 |
| --- | --- |
| `time` | `time_coverage.time_count` |
| `channel` | `channels` 数量 |
| `lat` | `grid.latitude.count` |
| `lon` | `grid.longitude.count` |
| `mask_channel` | `mask.channels` 数量 |

当前主数组布局为 `(time, channel, lat, lon)`，整帧分块为 `(1, C, H, W)`。附件中的 `channel` 长度按附件自己的通道表确定，不能套用主数组的 `C`。

`mask` 描述内嵌 `mask/mask_channel` 的语义与取值要求。`valid_value: 1` 表示属于该掩膜通道的区域，不表示气象值没有缺失；`complement_sum: 1` 要求每个网格点的陆地和海洋掩膜恰好互补。外部 mask 附件不在这组规则中。

### 3.5 属性声明与实际参数

`metadata` 组织 Zarr 属性要求，`normalization` 组织实际参数要求：

| 分组 | 对应检查 |
| --- | --- |
| `metadata.data_attributes` | 核对 `data.attrs` 中的数值表示、处理方式、参数位置和恢复公式声明 |
| `metadata.channel_metadata` | 核对 `channel.attrs` 中逐通道信息的位置、结构版本、完整覆盖、说明及缩放/偏移参数 |
| `normalization.statistics` | 核对 `mean/std` 数值有限，且每个 `std > 0` |
| `normalization.effective_statistics` | 核对特定变量的有效参数，例如 tp 使用 `mean=0`、`std=1` |

`channel.channel_info[*].scale_factor` 是项目约定的位置说明，表示各完整通道的 `channel.attrs["channel_info"][通道名]["scale_factor"]`；它不是自动执行的路径表达式。`scale_factor` 与 `std` 对应，`add_offset` 与 `mean` 对应。

公式字符串用于属性一致性检查，不作为代码执行。恢复公式先得到预处理值，不保证已经恢复来源单位。`normalized` 也不能被理解为每个通道都执行了 z-score；例如 tp 的实际处理是单位转换、截断负值和 log1p，其有效参数不是样本真实均值和标准差。

### 3.6 辅助文件

`auxiliaries.files` 按文件名登记要求。`format`、`role`、`required`、维度、dtype 和数值规则分别由各附件实际标准确定。

- 文件名相对于 `auxiliary_directory` 定位，必需文件缺失应报告失败。
- `grid: main_data` 引用本套 `meta_info.grid`；有坐标的附件比较其实际空间轴的长度、数值及顺序。
- 只有纬度轴的权重只比较纬度，不要求增加经度轴。
- NPY 没有坐标信息，只能直接核对轴长度；坐标归属仍需独立依据。
- `singleton` 表示长度为 1 的轴，不是要求 NPY 保存维度名称。
- NetCDF 内部数据变量名暂不约束；附件自身的 channel 标签及顺序仍可有明确要求。
- 未登记附件只报告警告，不代表其内容已经检查通过。
- 权重要求有限值、`weight >= -nonnegative_atol` 且至少有一个正值；容差内负值仍记录警告，不改写原值。

## 4. 第二部分：variable_metadata

每个键是基础变量名或完整单层变量名。逐变量使用单行 mapping，各条目之间不按类别插入空行。

| 字段 | 含义 |
| --- | --- |
| `units` | 预处理结果的单位，不直接代表主数组归一化数值的单位 |
| `long_name` | 变量的物理含义说明 |
| `source_name` | 来源变量名；派生变量可明确列出多个来源分量，不是来源文件路径 |
| `source_units` | 处理前的来源单位，允许与 `units` 不同 |
| `variable_type` | 已声明的变量处理类型，如瞬时、累积、派生或静态 |
| `preprocessing` | 有序处理步骤，操作及参数均须由程序明确支持 |
| `level_type` | 单层变量的类型，如 `height_above_ground`、`soil_layer`、`surface` |
| `level` | 固定高度或土壤层索引的数值 |
| `level_units` | 层值单位，与变量自身的 `units` 不同 |
| `unconfirmed_fields` | 明确未确认的字段清单，不填候选值充当已确认标准 |

固定高度变量保存自己的 `level_type/level/level_units`。土壤层的 `layer_index` 是层编号，不是统一米深度。地表变量只声明 `level_type: surface`，不增加虚构的零米层号。

气压层和海洋多深度变量的共享层次在第一部分定义，不在各变量中重复长层表。ORAS5 盐度的 `units` 与 `source_units` 保持 `null`，并明确标记未确认；它们不能因此被视为已通过单位验收。

当前产品处理步骤使用以下操作及参数，列表顺序就是处理顺序：

| operation | 含义 | 主要参数 |
| --- | --- | --- |
| `multiply` | 乘以转换系数 | `factor`、`input_units`、`output_units` |
| `divide` | 按明确除数转换，例如辐射累积量转平均通量 | `divisor`、`accumulation_window_seconds`、`input_units`、`output_units` |
| `clip_min` | 将低于下限的数值截断到下限 | `minimum` |
| `log1p` | 计算 `log(1+x)` | 当前无额外参数 |
| `hypot` | 根据两个明确来源分量计算模长 | `inputs` |
| `zscore` | 使用对应通道参数进行标准化 | `mean`、`std`，当前分别引用 `/mean`、`/std` |

例如，普通比湿先转换单位再做 z-score；当前 tp 做单位转换、非负截断和 log1p，不再做 z-score。不同产品辐射窗口可能不同，不能只按相同变量名复用处理参数。

## 5. 两部分的关联规则

以 CRA c71 的 `q850` 为例：

```text
meta_info.pressure.variables中的q
    + 合法层号850及声明的通道名模板
    → 完整通道q850，层值850 hPa

variable_metadata.q
    → 比湿含义、来源单位、目标单位及处理步骤

meta_info.channels
    → q850是否存在及其完整通道顺序
```

加载器应先检查变量、层次、完整通道之间的覆盖关系，再生成逐通道要求。每个通道必须有唯一解释；缺失、重复或冲突的关联应报错。没有层次定义的变量不能自动当作地表变量，也不能对所有数字后缀使用同一种解析规则。

通道元数据检查主要证明声明符合标准，不单凭 `source_name`、单位或层号字符串证明实际数值确实来自正确的来源和物理层次。

## 6. 各产品的适用差异

| 产品或配置 | 应保留的差异 |
| --- | --- |
| CRA c71 | 71 通道；主纬度南到北，气压变量共用本产品的 13 层 |
| CRA c87 | 87 通道；两套网格、时间覆盖和常量附件分别维护 |
| ERA5 c78 / c87 | 分别保留 78 / 87 通道；涉及辐射的处理窗口分别为 21600 / 86400 秒 |
| ERA5 c116 | 116 通道，含 10 hPa；保留分段排列及固定土壤层变量 |
| ERA5-Land c8 | 仅八个土壤通道，不声明气压层或 tp 特殊参数；只登记权重附件 |
| ORAS5 c88 | 使用存量 `os/ot/ou/ov`、显式深度标签及 NetCDF 权重；盐度单位未确认 |
| S2S c76 | 保留自身降序气压通道及 `tcwv/ttr` 等变量；辐射窗口为 86400 秒 |

除 CRA c71 外，当前其余九套的主纬度均为北到南。网格大小、附件通道及时间覆盖不能仅凭同名变量或通道数量推断。

## 7. 编写与维护约定

1. 先确定产品实际标准，再填写完整、有序的 `channels`，不得从被检数据自动回写期望值。
2. 时间间隔同时填写数值和单位，独立核对首末、数量和相位。
3. 网格同时声明首末、数量、间隔和方向，保留产品实际纬度方向。
4. 只填写适用共享层次，不为没有气压层的数据补充 pressure 模板。
5. 变量字典仅包含本产品使用的变量，补齐来源、单位、层次及处理步骤。
6. 按附件实际格式、轴结构和通道标签填写规则，不复制其他产品附件。
7. 保留未知字段的明确状态，不以默认单位或质量阈值补齐尚未确认的标准。
8. 注释使用简短、正式的行末说明；不在配置中记录开发过程或测试结论。

追加时间数据时主要更新该套时间覆盖及适用修订信息。新增通道或改变处理标准时，应同步更新完整通道表、变量字典及层次关联；与旧数据不兼容的变更应保留独立标准，不能覆盖仍在使用的定义。

## 8. 与合并文件的对应关系

| 独立文件中的内容 | 合并文件中的位置 |
| --- | --- |
| 数组、mask、属性、基础统计约束 | `meta_info.common` |
| 产品完整通道、共享层次及特殊有效参数 | `meta_info.products.<产品键>` |
| 本套模板版本、路径、时间、网格、数据版本、修订及附件 | `meta_info.datasets.<短配置标识>` |
| 本产品变量字典 | `variable_metadata.<产品键>` |

合并文件的产品键与第二部分产品键必须一一对应；数据条目的 `product` 同时关联两部分。独立文件已经选定一套数据，不再增加 common/products/datasets 三个内部注册分组，也不在变量字典外重复一层产品键。

以上对应关系仅说明未采用的历史方案。当前只维护独立文件，不从合并文件导出，
也不要求两套同步。该声明文件合并方案与跨数据集组合校验不是同一概念；
后者已从当前数据验收流程移除。

## 9. 程序接入与验证边界

独立 YAML 已接入严格加载器及统一验收入口，替代旧 JSON profile 和 Python 产品验收声明。
加载器检查字段合法性、显式产品关联、完整通道及层次展开，并将数组、坐标、属性、
统计参数、预处理、内嵌 mask 和附件规则交给现有通用检查器。
重复键、alias/merge、未知字段、未使用变量和不支持的布局/运算会明确报错。
固定标准模型和严格 YAML 加载位于 `datakit.schema.datasets`，与十份独立 YAML 一同打包；
`datasets/specifications.py` 是处理标准的程序结构，而非另一份产品标准。
具体数组、属性、掩膜和通道元数据不再由 Python 默认标准补齐；必填声明缺失时直接报错。
预期块形状从 YAML 的分块声明解析。扫描器仍仅支持整通道、整空间、单时次分块，
不支持的数组类型、分块及变换公式仍明确拒绝；删除默认标准不扩大执行能力。
运行绑定、对象检查、报告和执行位于 `pipelines.dataset_building` 的 checks、workflows 和 support。
生产定义单独位于 `pipelines.dataset_building.production`，不用于推导验收真值；
`examples.dataset_building` 只提供 CLI。
`datasets/specifications.py` 定义固定模型、约束和推导，`datasets/inputs.py` 严格加载 YAML；
`support/specifications.py` 定义运行和转换对照规格，`support/inputs.py` 装配执行 JSON。
原 acceptance 目录移除，`checks` 按格式、数值、存储和转换组织规则，
单套 quick/full 扫描和批量执行在 `workflows/dataset.py`，转换执行在 `workflows/conversion.py`。
`support/reports.py` 负责结果、完成结论汇总和保存；转换回执核验归入 `checks/conversion.py`。

标准 YAML 不接受 `snapshot_id`、`production_evidence`、`conversion_evidence`。
运行时绑定放在独立 JSON 的 `execution` 对象中，通过 `schema`
引用固定 YAML；转换专项 JSON 的 `target_schema` 也遵守此约定。
这样回执可绑定 YAML 原始字节摘要，而无需把回执引用写回 YAML 形成摘要循环。
执行配置不能覆盖目标规则或放宽已要求的转换门槛，执行源摘要独立保存并检查不变。
消费者、模型测试及相关回执不属于本数据验收流程；旧相关字段不再接受。
`grid: main_data` 由程序核对 NC 坐标；无坐标 NPY 仍须独立 `grid_provenance`，
不能以形状相同代替来源证明。公式仅支持明确定义的形式，不执行任意代码或自动转换数据。

配置核对通过不等于服务器数据已通过 quick/full，也不能替代全量扫描和适用的生产完成证据。现有条款及运行方法分别见 [校验计划](VALIDATION_PLAN.md) 和 [程序使用说明](../DATASET_VALIDATION.md)。
