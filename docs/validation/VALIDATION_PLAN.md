# 标准数据集校验计划

版本：1.5

生效日期：2026-10-13

适用模板版本：独立 YAML `v1`

校验范围标识：`confirmed-format`

声明组织修订：以十份独立 YAML 替代 JSON profile 和 Python 产品验收声明；保留已确认规则及扫描门槛。

## 1. 目的与适用范围

本计划用于验收独立生产或刷新后交付的标准数据集。验收对象为完整数据集的结构、
坐标、通道、已确认的处理声明、统计量、内嵌掩膜、已登记辅助结构及存储读取完整性。
本版只纳入已确认规则，不以暂定规则或默认经验阈值作验收依据。

当前十套实例以所有者指定的 refresh 数据集为格式基准。通道清单、顺序和网格已冻结在
各套独立 YAML 中；运行时不得从被检数据反推标准。新生产批次使用同一产品标准，另存
独立 YAML 冻结路径、时间范围和数据修订号，不将本批的起止时间套用于其他批次。

程序始终只读。不得自动改名、以 `level` 代替 `channel`、重排、翻转网格、填零、
裁剪负权重、修复元数据或用旧附件补齐主 Zarr。下游读取器的兼容行为不改变本版命名标准。
本流程不执行模型读取、训练或推理，不核验消费者及模型测试回执。
跨数据集的组合校验不属于本计划；多份标准由外围循环分别验收，不增加联合使用门槛。

## 2. 标准组织与职责

| 层次 | 位置 | 内容 |
|---|---|---|
| 固定目标模型 | `datakit/schema/specifications.py` | 标准结构、声明约束和期望值推导，不包含运行快照或回执绑定 |
| 标准加载 | `datakit/schema/parser.py` | 严格 YAML 加载、关联校验、路径及原始字节摘要 |
| 运行绑定 | `pipelines/dataset/configuration.py` | 运行快照/回执及转换对照规格，不回写 YAML |
| 执行输入 | `pipelines/dataset/configuration.py` | 调用基础标准加载器、装配执行 JSON、执行源摘要及独立证据绑定 |
| 独立标准 | `config/dataset_specs/*.yaml` | 完整有序通道、可验收语义字段、网格、掩膜、辅助结构和批次信息 |
| 生产定义与元数据 | `pipelines/dataset/old/raw_truth.py`、`old/channel_metadata.py` | c116 来源映射及生产元数据生成；不从生成器推导验收真值 |
| 严格 IO | `datakit/validation/zarr_inspection.py` | 原始元数据、指定块、物理块键、编码支持范围 |
| 条目检查 | `datakit/validation/structure.py`、`values.py`、`storage.py` 及 pipeline 的 `evidence.py` | 主数据/附件格式与数值、物理完整性、转换对照及回执业务规则 |
| 单套验收与扫描 | `pipelines/dataset/validate.py` | 单套 quick/full 主块读取、物理遍历、有界进度、中断及资源控制 |
| 转换对照 | `pipelines/dataset/old/conversion_validation.py` | 转换保真 quick/full，配置装配调用 inputs，规则调用 checks |
| 报告 | `pipelines/dataset/reports.py` | 完成结论汇总、输出保护和报告保存；不核验转换回执业务内容 |
| 执行入口 | `pipelines.dataset` 的公共 Python API；`examples/dataset_building/history/validate_standard_conversion.py` | Python 调用与转换对照命令行适配 |

基础库提供固定标准模型、严格 YAML 解析并随包携带 YAML；运行绑定、检查、证据及执行集中在流程层。
旧目录层级已移除，基础检查位于 datakit.validation，pipeline 按操作、证据、配置和报告平铺。
依赖方向为 workflows → checks → support，验收不导入生产定义，规则不自行打开或遍历文件。
基础库不反向依赖 pipeline 或模型项目。每份标准自包含，不使用合并草稿、alias/merge 或 Python 产品注册表。
配置标识、根 ID、产品版本、模板版本、数据版本及 SHA256 分开记录。
`status: approved` 只表示已纳入规则获准使用，不表示数据已通过 full 或其他用途验收。
ERA5 c116 的来源映射及通道元数据生成仍用于生产，集中在 `old/raw_truth.py` 和 `old/channel_metadata.py`，不从生成器推导验收真值。

## 3. 校验条款

### A. 身份、目录与声明

1. **A01** 根 `dataset_id`、`schema_version=2.0`、`content_version=v3`、`data_revision` 与实例一致。
2. **A02** 目录名与冻结值一致；目录名和根 ID 不同须有显式实例映射，时间范围以完整时间轴为准。
3. **A03** 根空间范围与产品网格一致；主数组具备规范化表示、统计量引用和逆变换声明。
4. **A04** YAML 加载严格拒绝重复键、未知配置键和未支持声明；检查前后原始字节 SHA256 相同，记录产品、配置、模板、掩膜和工具版本。

### B. 主 Zarr 结构与存储编码

1. **B01** 本批使用 Zarr v3，根数组准确为 `data/time/channel/lat/lon/mean/std/mask/mask_channel`；不得有嵌套主数据组或额外主数组。
2. **B02** `data` 维度准确为 `(time, channel, lat, lon)`，shape 与冻结实例一致，原始 dtype 与 YAML 一致；当前十套均为 `float16`，程序也支持显式 `float32`。
3. **B03** 主块准确为 `(1, C, H, W)`。块边界从冻结 shape/chunks 计算，先确定预期数量和索引范围，再访问数据。
4. **B04** 其余数组维度、shape、原始 dtype 符合下表；禁止隐式转换后再判断。
5. **B05** v3 `dimension_names` 与读取对象一致；若同时存在旧 `_ARRAY_DIMENSIONS`，二者必须一致。
6. **B06** consolidated metadata 存在时逐项与实际子数组元数据比较；不得只信合并缓存。
7. **B07** 编码须能严格解码，物理覆盖须处于已支持范围。当前支持本地非分片目录、可定位的 v2/default 块键编码；不支持的分片或后端报告 `blocked`。

| 数组 | 维度 | shape | 原始 dtype |
|---|---|---|---|
| data | time, channel, lat, lon | T,C,H,W | float16 |
| time | time | T | datetime64[ns] |
| channel | channel | C | string |
| lat / lon | lat / lon | H / W | float32 |
| mean / std | channel | C | float32 |
| mask | mask_channel, lat, lon | 2,H,W | uint8 |
| mask_channel | mask_channel | 2 | string |

### C. 完整时间轴

1. **C01** 即使 quick，也读取并校验完整时间坐标；禁止只验证被抽样时刻。
2. **C02** 原始 dtype 为纳秒时间，无 NaT、重复、逆序或乱序。
3. **C03** 按独立冻结的首值、末值、频率生成预期轴，逐值完全相等，数量独立核对；缺点、额外点及同数量替换均失败。
4. **C04** `timezone=UTC`；h6 为 00/06/12/18 UTC，h24 为 00 UTC。按日历生成轴，闰日自动包括在内。

### D. 要素通道及已确认处理声明

1. **D01** `channel` 名称、数量和顺序与产品完整清单逐项一致；同数量错通道或错序失败。
2. **D02** `channel_info` 为完整逐通道字典，schema 版本为 `1.0`，每个通道有非空描述。
3. **D03** 按产品可验收字段检查来源名称、层次类型/层号/层单位、变量类型、处理步骤及其参数；不会通用拆数字后缀生成标准。
4. **D04** 气压层、土壤层、近地高度、日极值和海洋深度分别按产品显式定义解释；不同产品版本不自动使用别名或替换深度代码。
5. **D05** q/q2m 声明 kg/kg 到 g/kg 的乘 1000 与 zscore；tp 声明 m 到 mm、clip_min(0)、log1p，有效 mean/std 为 0/1。
6. **D06** 辐射声明除以产品固定累积秒数再 zscore；风速声明对应 u/v 分量的 hypot；其他声明依产品定义检查。
7. **D07** 声明合规只证明声明与标准一致，不据此推断物理数值真实性。

### E. 经纬度网格

1. **E01** quick/full 均完整读取经纬度；shape、dtype、每个坐标值和方向与产品网格一致。
2. **E02** p25 为 721×1440，间隔 0.25°；1p5 为 121×240，间隔 1.5°。
3. **E03** 经度东向递增，自 0 起；分别止于 359.75°、358.5°，不得重复 360°。
4. **E04** CRA c71 纬度自 −90° 到 +90°；其余九套自 +90° 到 −90°。不得靠自动翻转通过。

### F. 统计量与逆变换一致性

1. **F01** 主 Zarr 自带一维 `mean/std(channel)`，数量 C，原始 float32；不以根属性或外部 sidecar 代替。
2. **F02** mean/std 全部有限，std 全部大于 0；tp 有效 mean=0/std=1。
3. **F03** 逐通道 `scale_factor=std`、`add_offset=mean`，数值类型正确且逐项相等。
4. **F04** 声明逆变换 `preprocessed_value = normalized_value * scale_factor + add_offset`，引用与主数组属性一致。

### G. 产品内嵌掩膜

1. **G01** 规则归属每个产品 schema 的 `MASK`，共享 `xmetai.land-sea-mask/1.0`。
2. **G02** 主 Zarr 中 `mask` 和 `mask_channel` 符合 B04；标签顺序准确为 `land_mask, sea_mask`。
3. **G03** 完整掩膜只含 0/1；相同像素两通道相加为 1。用足够宽的整数运算检查，防止 uint8 溢出掩盖异常。
4. **G04** 掩膜与主数据共享 lat/lon 网格。quick 也完整检查掩膜，不抽样。

### H. 辅助文件结构与权重

1. **H01** 各套规定的 const、weight 必须存在、可只读打开；校验原始维度、shape、dtype、坐标及已声明的 const 通道顺序。
2. **H02** NPY 禁用 pickle，并有已确认的网格关联；NC 依结构选择数据变量。没有唯一可定位变量时警告并记录未执行的值检查，不猜测变量。
3. **H03** 已选辅助数组值全部有限。权重至少有一个正值，全部满足 `weight >= -1e-7`；容差内微小负值记录 warning，不裁剪原值。低于界限失败。
4. **H04** 未登记 NC/NPY 或 NC 中额外数据变量作为 warning；不将其自动认定为已批准角色，也不因此否定主数据。
5. **H05** 外部 mask 附件不参与本版验收，程序在打开前即标记 `not-applicable`；掩膜验收使用 G 类条款。
6. **H06** 其他未声明根文件/子目录及额外主 Zarr 数组仍属于失败，不使用辅助警告规则绕过 B01。

### I. 原始主数据块读取与检查

1. **I01** 指定块严格只读；保留原始 shape/dtype/编码和异常，禁止把读取异常吞成 None、替换成填充值或使用下游适配后值。
2. **I02** 每个已选块读回 shape 必须与逻辑块边界一致，原始 dtype 为 float16。
3. **I03** 原始所有通道不得有正负 Inf；tp 的有限存储值必须非负，符合已确认 clip_min(0)→log1p 处理域。
4. **I04** 记录每通道已检值数、NaN/Inf 数、有限值比例、有限值最小/最大值和常数块数；诊断统计不另行产生阈值判定。
5. **I05** 一次最多持有一个主块和有界逐通道统计状态；先校验内存预算。p25 c116 的 float16 整帧约 230 MiB，另为解码和临时计算留出预算。

### J. 物理覆盖、稳定性及完成证据

1. **J01** 坐标、统计量、掩膜等小数组在 quick/full 均检查全部预期物理块键；成功返回 fill_value 不等于存在块。
2. **J02** quick 核查被抽样主块的物理键；full 核查全部预期主块，且核查主数组目录无意外块键。
3. **J03** 分别统计成功读取、读取失败、未扫描，以及物理存在、经规则允许省略、未批准缺失、存储未知；两套计数分别与预期总数守恒。
4. **J04** 本批十份 YAML 无允许省略块。日后采用稀疏省略时，须提前显式列出逻辑索引并绑定独立生产证明；不能从实际缺块倒推规则。
5. **J05** 扫描前后元数据摘要一致，逐块检查文件指纹变化；发现变化失败。
6. **J06** full 完成还须有独立冻结的生产记录，绑定 dataset_id、data_revision、snapshot_id、shape、预期块数、允许省略表、complete=true 及 writes-stopped/immutable-snapshot 状态，并通过预先冻结的 SHA256。
7. **J07** quick 不要求 J06 的生产证明来判定抽样格式通过；缺失证明单列为未执行的发布条件，不能用于允许缺块。
8. **J08** 未读完、读取异常、物理缺失、不支持编码、中断、证据不足时不得汇总为 full pass。

### K. 报告和结论

1. **K01** 报告绑定 dataset_id、产品/配置/模板/掩膜/工具版本、data_revision、路径、YAML SHA256、run_id、模式和已检范围；验收报告及转换回执不得跨标准复用。
2. **K02** 指定 output 时保存一个 JSON，记录逐项状态、位置、依据、覆盖数、连续完成范围、耗时和内存；逐块记录由外围通过 on_block 回调保存。
3. **K03** `pass` 为已执行通过；`fail` 为违反已纳入规则；`warning` 不阻断；`not-applicable` 为范围外；`not-run/blocked` 不能被描述为已通过。
4. **K04** `quick_passed=true` 要求完成预定抽样、各项适用 quick 检查通过且未中断。不能因尚无 fail 就认定完成；max-blocks 截断 quick 也不通过。
5. **K05** `scan_complete=true` 要求 full 的值检查、物理覆盖、计数一致、稳定证明全部完成；该字段区分“扫描完成”与“扫描中发现违规”。
6. **K06** `release_passed=true` 要求 full 完成且全部本版适用规则无 fail/blocked/needs-review/not-run。此结论限于本版范围，不能扩大成所有模型或所有科学用途的证明。
7. **K07** 调用方读取报告的 quick_passed、release_passed 及具体检查状态；配置或输出错误抛出异常。quick 通过不表示 full 已运行。

## 4. quick 与 full

| 范围 | quick | full |
|---|---|---|
| 完整时间轴、通道、经纬度、统计量、掩膜、已登记辅助结构 | 全部检查 | 全部检查 |
| 主数据块 | 默认首、中、末三个预期块 | 全部预期块 |
| 主数据物理覆盖 | 仅所选块 | 全部块及意外块键 |
| 原值检查及诊断 | 所选块 | 全部块 |
| 生产冻结证明 | 单独记录，不阻断非稀疏抽样 | 必须具备 |
| 未抽样块结论 | 不作通过判定 | 全部已检后给出结论 |
| 扫描完成/发布标志 | 始终 false | 按 J/K 条款判定 |

默认抽样索引为 `linspace(0,T-1,3)` 取整并去重；报告写出实际索引。增大
`--sample-blocks` 只扩大 quick 样本。`--mode full` 且不传 `--max-blocks` 才会尝试全量扫描。
扫描中断仍保存部分报告和未扫描数量，不保证达到 full。

## 5. 各套数据标准

下列每套独立冻结实例，均适用 A—K 条款。产品有序通道表为逐项严格对照表，
不按字母排序；原始格式基准与被检批次分开维护。主块均为一个时间索引的完整 C×H×W 帧。

### 5.1 cra.c71 / p25.h6

- 产品标准：`cra.c71/1.0`；独立标准：`cra_c71_p25_h6.yaml`。
- 根 ID：`cra.1979_2025.c71.p25.h6.v1`；目录：`cra.19790101_20260714.c71.p25.h6.v1`；修订号：`20260921`。
- 时间：`1979-01-01T00:00:00.000000000` 至 `2026-07-14T18:00:00.000000000`，UTC 每 6 小时，共 69,448 个时刻。
- 主 shape：`(69448, 71, 721, 1440)`；chunk：`(1, 71, 721, 1440)`。
- 纬度：`-90 → 90`；经度：`0 → 359.75`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`orography, lsm, coslat, sinlat, coslon, sinlon`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。

完整 71 通道顺序：

```text
z50, z100, z150, z200, z250, z300, z400, z500, z600, z700,
z850, z925, z1000, t50, t100, t150, t200, t250, t300, t400,
t500, t600, t700, t850, t925, t1000, u50, u100, u150, u200,
u250, u300, u400, u500, u600, u700, u850, u925, u1000, v50,
v100, v150, v200, v250, v300, v400, v500, v600, v700, v850,
v925, v1000, q50, q100, q150, q200, q250, q300, q400, q500,
q600, q700, q850, q925, q1000, t2m, q2m, u10m, v10m, msl,
tp
```

### 5.2 cra.c87 / p25.h24

- 产品标准：`cra.c87/1.0`；独立标准：`cra_c87_p25_h24.yaml`。
- 根 ID：`cra.1979_2024.c87.p25.h24.v1`；目录：`cra.1979_2024.c87.p25.h24.v1`；修订号：`20260920`。
- 时间：`1979-01-01T00:00:00.000000000` 至 `2024-12-31T00:00:00.000000000`，UTC 每 24 小时，共 16,802 个时刻。
- 主 shape：`(16802, 87, 721, 1440)`；chunk：`(1, 87, 721, 1440)`。
- 纬度：`90 → -90`；经度：`0 → 359.75`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`geo, lsm, coslat, sinlat, coslon, sinlon`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。
- 辐射累积窗：`[86400]` 秒；声明与此严格一致。

完整 87 通道顺序：

```text
z50, z100, z150, z200, z250, z300, z400, z500, z600, z700,
z850, z925, z1000, t50, t100, t150, t200, t250, t300, t400,
t500, t600, t700, t850, t925, t1000, u50, u100, u150, u200,
u250, u300, u400, u500, u600, u700, u850, u925, u1000, v50,
v100, v150, v200, v250, v300, v400, v500, v600, v700, v850,
v925, v1000, q50, q100, q150, q200, q250, q300, q400, q500,
q600, q700, q850, q925, q1000, msl, t2m, t2m_min, t2m_max, d2m,
sst, ws10m, ws100m, u10m, v10m, u100m, v100m, lcc, mcc, hcc,
tcc, ssr, ssrd, fdir, ttr, tcw, tp
```

### 5.3 cra.c87 / 1p5.h24

- 产品标准：`cra.c87/1.0`；独立标准：`cra_c87_1p5_h24.yaml`。
- 根 ID：`cra.1979_2025.c87.1p5.h24.v1`；目录：`cra.1979_2025.c87.1p5.h24.v1`；修订号：`20260921`。
- 时间：`1979-01-01T00:00:00.000000000` 至 `2025-12-31T00:00:00.000000000`，UTC 每 24 小时，共 17,167 个时刻。
- 主 shape：`(17167, 87, 121, 240)`；chunk：`(1, 87, 121, 240)`。
- 纬度：`90 → -90`；经度：`0 → 358.5`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`z, lsm`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。
- 辐射累积窗：`[86400]` 秒；声明与此严格一致。

完整 87 通道顺序：

```text
z50, z100, z150, z200, z250, z300, z400, z500, z600, z700,
z850, z925, z1000, t50, t100, t150, t200, t250, t300, t400,
t500, t600, t700, t850, t925, t1000, u50, u100, u150, u200,
u250, u300, u400, u500, u600, u700, u850, u925, u1000, v50,
v100, v150, v200, v250, v300, v400, v500, v600, v700, v850,
v925, v1000, q50, q100, q150, q200, q250, q300, q400, q500,
q600, q700, q850, q925, q1000, msl, t2m, t2m_min, t2m_max, d2m,
sst, ws10m, ws100m, u10m, v10m, u100m, v100m, lcc, mcc, hcc,
tcc, ssr, ssrd, fdir, ttr, tcw, tp
```

### 5.4 era5.c87 / p25.h24

- 产品标准：`era5.c87/1.0`；独立标准：`era5_c87_p25_h24.yaml`。
- 根 ID：`era5.1959_2025.c87.p25.h24.v1`；目录：`era5.1959_2025.c87.p25.h24.v1`；修订号：`20260922`。
- 时间：`1959-01-01T00:00:00.000000000` 至 `2025-12-31T00:00:00.000000000`，UTC 每 24 小时，共 24,472 个时刻。
- 主 shape：`(24472, 87, 721, 1440)`；chunk：`(1, 87, 721, 1440)`。
- 纬度：`90 → -90`；经度：`0 → 359.75`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`geo, lsm, coslat, sinlat, coslon, sinlon`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。
- 辐射累积窗：`[86400]` 秒；声明与此严格一致。

完整 87 通道顺序：

```text
z50, z100, z150, z200, z250, z300, z400, z500, z600, z700,
z850, z925, z1000, t50, t100, t150, t200, t250, t300, t400,
t500, t600, t700, t850, t925, t1000, u50, u100, u150, u200,
u250, u300, u400, u500, u600, u700, u850, u925, u1000, v50,
v100, v150, v200, v250, v300, v400, v500, v600, v700, v850,
v925, v1000, q50, q100, q150, q200, q250, q300, q400, q500,
q600, q700, q850, q925, q1000, msl, t2m, t2m_min, t2m_max, d2m,
sst, ws10m, ws100m, u10m, v10m, u100m, v100m, lcc, mcc, hcc,
tcc, ssr, ssrd, fdir, ttr, tcw, tp
```

### 5.5 era5.c78 / p25.h6

- 产品标准：`era5.c78/1.0`；独立标准：`era5_c78_p25_h6.yaml`。
- 根 ID：`era5.1979_2024.c78.p25.h6.v1.xujn`；目录：`era5.1979_2024.c78.p25.h6.v1`；修订号：`20260921`。
- 时间：`1979-01-01T00:00:00.000000000` 至 `2024-12-31T18:00:00.000000000`，UTC 每 6 小时，共 67,208 个时刻。
- 主 shape：`(67208, 78, 721, 1440)`；chunk：`(1, 78, 721, 1440)`。
- 纬度：`90 → -90`；经度：`0 → 359.75`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`orography, lsm, coslat, sinlat, coslon, sinlon`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。
- 辐射累积窗：`[21600]` 秒；声明与此严格一致。

完整 78 通道顺序：

```text
z50, z100, z150, z200, z250, z300, z400, z500, z600, z700,
z850, z925, z1000, t50, t100, t150, t200, t250, t300, t400,
t500, t600, t700, t850, t925, t1000, u50, u100, u150, u200,
u250, u300, u400, u500, u600, u700, u850, u925, u1000, v50,
v100, v150, v200, v250, v300, v400, v500, v600, v700, v850,
v925, v1000, q50, q100, q150, q200, q250, q300, q400, q500,
q600, q700, q850, q925, q1000, t2m, d2m, sst, u10m, v10m,
u100m, v100m, msl, ssr, ssrd, fdir, ttr, tp
```

### 5.6 era5.c116 / p25.h6

- 产品标准：`era5.c116/1.0`；独立标准：`era5_c116_p25_h6.yaml`。
- 根 ID：`era5.202501_202607.c116.p25.h6.v3`；目录：`era5.202501_202607.c116.p25.h6.v3`；修订号：`20260917`。
- 时间：`2025-01-01T00:00:00.000000000` 至 `2026-07-31T18:00:00.000000000`，UTC 每 6 小时，共 2,308 个时刻。
- 主 shape：`(2308, 116, 721, 1440)`；chunk：`(1, 116, 721, 1440)`。
- 纬度：`90 → -90`；经度：`0 → 359.75`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`orography, lsm, coslat, sinlat, coslon, sinlon`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。
- 辐射累积窗：`[21600]` 秒；声明与此严格一致。

完整 116 通道顺序：

```text
z1000, z925, z850, z700, z600, z500, z400, z300, z250, z200,
z150, z100, z50, t1000, t925, t850, t700, t600, t500, t400,
t300, t250, t200, t150, t100, t50, u1000, u925, u850, u700,
u600, u500, u400, u300, u250, u200, u150, u100, u50, v1000,
v925, v850, v700, v600, v500, v400, v300, v250, v200, v150,
v100, v50, q1000, q925, q850, q700, q600, q500, q400, q300,
q250, q200, q150, q100, q50, d2m, t2m, msl, sp, skt,
u10m, v10m, u100m, v100m, sst, tcw, tp, sd, z10, t10,
u10, v10, q10, w1000, w925, w850, w700, w600, w500, w400,
w300, w250, w200, w150, w100, w50, w10, ws10m, ws100m, lcc,
mcc, hcc, tcc, ssr, ssrd, fdir, ttr, swvl1, swvl2, stl1,
stl2, wmb, mwd, cdww, mwp, swh
```

### 5.7 era5-land.c8 / 1p5.h24

- 产品标准：`era5-land.c8/1.0`；独立标准：`era5_land_c8_1p5_h24.yaml`。
- 根 ID：`land.1950_2024.c8.1p5.h24.v1`；目录：`land.1950_2024.c8.1p5.h24.v1`；修订号：`20260920`。
- 时间：`1950-01-01T00:00:00.000000000` 至 `2024-12-31T00:00:00.000000000`，UTC 每 24 小时，共 27,394 个时刻。
- 主 shape：`(27394, 8, 121, 240)`；chunk：`(1, 8, 121, 240)`。
- 纬度：`90 → -90`；经度：`0 → 358.5`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。

完整 8 通道顺序：

```text
stl1, stl2, stl3, stl4, swvl1, swvl2, swvl3, swvl4
```

### 5.8 oras5.c88 / 1p5.h24

- 产品标准：`oras5.c88/1.0`；独立标准：`oras5_c88_1p5_h24.yaml`。
- 根 ID：`oras5.1993_2024.c88.1p5.h24.v1`；目录：`oras5.1993_2024.c88.1p5.h24.v1`；修订号：`20260923`。
- 时间：`1993-01-01T00:00:00.000000000` 至 `2024-12-31T00:00:00.000000000`，UTC 每 24 小时，共 11,688 个时刻。
- 主 shape：`(11688, 88, 121, 240)`；chunk：`(1, 88, 121, 240)`。
- 纬度：`90 → -90`；经度：`0 → 358.5`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`z, lsm`。
- 辅助 `weight.nc`：netcdf，维度 `('lat',)`，dtype `float32`，角色 `weight`。

完整 88 通道顺序：

```text
os50, os266, os514, os809, os1177, os1652, os2275, os3087, os4118, os5385,
os6902, os8692, os10803, os13307, os16316, os19978, os22114, os27135, os33386, os41179,
os50863, ot50, ot266, ot514, ot809, ot1177, ot1652, ot2275, ot3087, ot4118,
ot5385, ot6902, ot8692, ot10803, ot13307, ot16316, ot19978, ot22114, ot27135, ot33386,
ot41179, ot50863, ou50, ou266, ou514, ou809, ou1177, ou1652, ou2275, ou3087,
ou4118, ou5385, ou6902, ou8692, ou10803, ou13307, ou16316, ou19978, ou22114, ou27135,
ou33386, ou41179, ou50863, ov50, ov266, ov514, ov809, ov1177, ov1652, ov2275,
ov3087, ov4118, ov5385, ov6902, ov8692, ov10803, ov13307, ov16316, ov19978, ov22114,
ov27135, ov33386, ov41179, ov50863, sit, ssh, sic, mlt
```

### 5.9 oras5.c88 / p25.h24

- 产品标准：`oras5.c88/1.0`；独立标准：`oras5_c88_p25_h24.yaml`。
- 根 ID：`oras5.1993_2024.c88.p25.h24.v1`；目录：`oras5.1993_2024.c88.p25.h24.v1`；修订号：`20260923`。
- 时间：`1993-01-01T00:00:00.000000000` 至 `2024-12-31T00:00:00.000000000`，UTC 每 24 小时，共 11,688 个时刻。
- 主 shape：`(11688, 88, 721, 1440)`；chunk：`(1, 88, 721, 1440)`。
- 纬度：`90 → -90`；经度：`0 → 359.75`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`geo, lsm, coslat, sinlat, coslon, sinlon`。
- 辅助 `weight.nc`：netcdf，维度 `('lat',)`，dtype `float32`，角色 `weight`。

完整 88 通道顺序：

```text
os50, os266, os514, os809, os1177, os1652, os2275, os3087, os4118, os5385,
os6902, os8692, os10803, os13307, os16316, os19978, os22114, os27135, os33386, os41179,
os50863, ot50, ot266, ot514, ot809, ot1177, ot1652, ot2275, ot3087, ot4118,
ot5385, ot6902, ot8692, ot10803, ot13307, ot16316, ot19978, ot22114, ot27135, ot33386,
ot41179, ot50863, ou50, ou266, ou514, ou809, ou1177, ou1652, ou2275, ou3087,
ou4118, ou5385, ou6902, ou8692, ou10803, ou13307, ou16316, ou19978, ou22114, ou27135,
ou33386, ou41179, ou50863, ov50, ov266, ov514, ov809, ov1177, ov1652, ov2275,
ov3087, ov4118, ov5385, ov6902, ov8692, ov10803, ov13307, ov16316, ov19978, ov22114,
ov27135, ov33386, ov41179, ov50863, sit, ssh, sic, mlt
```

### 5.10 s2s.c76 / 1p5.h24

- 产品标准：`s2s.c76/1.0`；独立标准：`s2s_c76_1p5_h24.yaml`。
- 根 ID：`s2s.1950_2026.c76.1p5.h24.v1`；目录：`s2s.19500101_20260417.c76.1p5.h24.v1`；修订号：`20260921`。
- 时间：`1950-01-01T00:00:00.000000000` 至 `2026-04-17T00:00:00.000000000`，UTC 每 24 小时，共 27,866 个时刻。
- 主 shape：`(27866, 76, 121, 240)`；chunk：`(1, 76, 121, 240)`。
- 纬度：`90 → -90`；经度：`0 → 358.5`。
- 内嵌掩膜：`xmetai.land-sea-mask/1.0`，标签 `land_mask, sea_mask`，完整二值互补。
- 辅助 `const.nc`：netcdf，维度 `('channel', 'lat', 'lon')`，dtype `float32`，角色 `const`。 通道顺序：`z, lsm`。
- 辅助 `weight.npy`：npy，维度 `('singleton', 'lat', 'singleton')`，dtype `float32`，角色 `weight`。
- 辐射累积窗：`[86400]` 秒；声明与此严格一致。

完整 76 通道顺序：

```text
z1000, z925, z850, z700, z600, z500, z400, z300, z250, z200,
z150, z100, z50, t1000, t925, t850, t700, t600, t500, t400,
t300, t250, t200, t150, t100, t50, u1000, u925, u850, u700,
u600, u500, u400, u300, u250, u200, u150, u100, u50, v1000,
v925, v850, v700, v600, v500, v400, v300, v250, v200, v150,
v100, v50, q1000, q925, q850, q700, q600, q500, q400, q300,
q250, q200, q150, q100, q50, t2m, d2m, sst, ttr, u10m,
v10m, u100m, v100m, msl, tcwv, tp
```

## 6. 验证与执行

### 6.1 程序回归

使用独立构造的小型 Zarr v3 和故障注入检查正常通过、时间缺点/重复/错相位、
通道错序/同数量替换、纬度反向、错误统计量和处理参数、掩膜格式/标签/互补、
辅助结构、权重容差边界、未抽样块异常、删除/损坏块、fill_value 掩盖、合法稀疏省略、
合并元数据不一致、内存限制和中断。另验证 quick 不能把未完成抽样当作通过，
未设阈值的诊断不能伪装成物理质量判定。

```bash
python -m pytest tests/dataset_building/acceptance -q
```

### 6.2 单套调用

```python
from pipelines.dataset import validate_zarr

report = validate_zarr(
    "/data/land.zarr",
    "config/dataset_specs/era5_land_c8_1p5_h24.yaml",
    mode="quick",
)
```

默认完整扫描；需要生产完成证明时，在外围通过 load_validation_run 绑定回执后传入 spec。
批量循环由外围实现。数据集验收使用 Python API。

### 6.3 输出与读取

`output=None` 只返回报告，不写文件。指定 output 为单个 JSON 文件时才保存，已有文件拒绝覆盖。
报告必须位于输入数据和附件目录之外，且不能覆盖配置或证据。JSON 包含逐项检查、覆盖统计及完成状态。
历史报告不会随新规则更新，重跑保存需另选文件。本计划与实测结果分开保存。
