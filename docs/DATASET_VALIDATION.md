# 标准数据集校验

当前规范见 [标准数据集校验计划](validation/VALIDATION_PLAN.md)。十份独立 YAML 位于
[`config/dataset_specs`](../config/dataset_specs)，使用模板 `v1` 和
`confirmed-format` 范围，仅验收已确认规则。每份文件独立声明身份、产品版本、时间、
通道、网格、预处理、统计量、内嵌掩膜和附件；不依赖合并文件或 Python 产品注册表。
主数据输入始终只读，其他批次应另存 YAML，不从被检数据反推期望值。

旧十份 JSON profile、inventory 和纯验收产品声明已被替代。JSON 仍用于报告、独立基准、
生产/转换证据及独立执行配置，不进行无差别清理。固定标准模型和严格 YAML 解析位于
`datakit.schema`；检查、扫描、运行绑定、证据和报告归入
`pipelines.dataset` 的 `checks`、`workflows`、`support`。
基础库不反向依赖流程、生产定义或模型项目。

本流程只负责数据构建后的格式、完整性、质量和转换保真，不执行外部
core 读取探测或模型测试，也不核验消费者/模型回执。工具版本 `1.7` 删除相关入口、
配置字段和 JSON 的消费者兼容栏目；历史报告不改写。
旧 `consumer_required`、`consumer_evidence`、`model_evidence` 配置会明确报错，不能静默忽略。
工具版本 `1.8` 进一步移除跨数据集组合校验、CLI 和配置模板，以及报告中的
`cross_dataset_compatibility` 分类和 `combination.scope` 占位项。
一次选择多份 YAML 仍会分别验收各套数据，但不判断其联合使用条件，也不要求各套时间或网格相同。

## 模块职责与流程入口

| 位置 | 职责 |
|---|---|
| `datakit/schema/specifications.py` | 固定标准结构、声明约束和期望值推导；包含 `DatasetSchema` |
| `datakit/schema/parser.py` | 严格 YAML 加载、关联检查、路径、原始字节摘要及标准源变更检查 |
| `pipelines/dataset/configuration.py` | DatasetValidationSpec、运行配置加载、快照/回执绑定和执行源摘要 |
| `config/dataset_specs/*.yaml` | 十套具体产品验收配置，由仓库配置目录维护，不随基础库打包 |
| `pipelines/dataset/old/raw_truth.py`、`old/channel_metadata.py` | c116 生产常量、来源映射及生产通道元数据生成；不改变生产算法 |
| `datakit/validation/zarr_inspection.py` | 严格只读元数据、原始块及物理键定位 |
| `datakit/validation/structure.py`、`values.py` | 主数据和附件的格式/语义、数值规则及有界逐通道统计 |
| `datakit/validation/storage.py` | 编码、物理覆盖及冻结证明规则；不自行读取文件 |
| `evidence.py` | 转换映射、统计量、数值误差和回执核验规则 |
| `validate.py` | 单套 quick/full，共用扫描、物理遍历、进度、中断和资源控制 |
| `old/conversion_validation.py` | 源—目标转换对照、配置驱动运行和逐块记录 |
| `reports.py` | 报告对象、quick/full 结论汇总与输出保护；JSON 由 validate_zarr 保存 |
| `examples/dataset_building/` | 制作示例及转换专项入口；数据集验收直接调用 validate_zarr |
| `tests/dataset_building/acceptance/` | 按扫描、布局、产品、质量、掩膜、辅助、确认范围、对照和报告分组的回归 |

```python
from pipelines.dataset import validate_zarr
from datakit.schema import load_schema

spec, locations = load_schema("config/dataset_specs/era5_land_c8_1p5_h24.yaml")
report = validate_zarr(locations["dataset_path"], spec, output=None)
```

流程入口接收 YAML 路径、通用 schema 或包含运行证据的实例，并返回报告对象。
`output=None` 不保存文件；指定 output 文件路径时保存单份 JSON，已有文件拒绝覆盖。
默认完整检查；抽样检查显式传 `mode="quick"`。数据集验收使用 Python API。
批量循环与汇总由外围实现。
验收内部类型可由 Python 调用方显式构造，但不再维护第二套内置产品标准。
固定标准组件集中在 `datakit.schema`，运行状态及执行机制留在工作流内部。
`datakit/schema/specifications.py` 是程序模型及计算逻辑，YAML 才是产品标准；两者不是同一标准的双份维护。
基础数据检查集中在 `datakit.validation`，`configuration.py` 装配执行输入，pipeline 负责流程、证据与任务报告。
`load_schema` 返回独立目标模型，`load_validation_schema` 包装为未绑定回执的运行规格，
`load_validation_run` 装配执行 JSON；它们共用同一份 YAML 解析，不混写标准与运行字段。
单套 quick/full 的扫描和资源辅助统一在 `validate.py`，运行器并入对应工作流。
附件布局归入 format，附件数值归入 values；转换回执业务规则归入 `evidence.py`。
依赖为 `workflows → checks → support → datakit`，流程也可直接使用支撑及基础标准；验收不导入生产定义，
支撑不反向导入规则或流程。旧内部路径不保留转发副本，保持检查规则与报告结论语义。
表中省略 `pipelines/dataset/` 的路径均相对该流程目录。
转换专项与单套校验共享证据和报告类型，保留各自入口；组合校验入口不再提供。
ORAS5 验收仅登记目标数据目录 `/workspace/data/zarr_refresh_v3` 中已有的 `oras5.c88/1.0`：
`os/ot/ou/ov` 通道、深度编码 `19978`（199.78 m），以及 1.5°、0.25° 两种日频网格。
不再登记 `so/to/uo/vo`、深度编码 `19979` 的产品 `2.0`，也不自动重命名或修改深度。
原始资料下载和 [ORAS5 转换示例](../examples/dataset_building/history/oras5_nc_to_zarr_88.py) 仍可使用
`so/to/uo/vo`；其生产元数据生成能力保留，但不因此成为上述目标数据的验收标准。
ERA5 c116 的生产 `CHANNELS`、来源目录/别名/变量映射及目标坐标继续保留在
`pipelines/dataset/old/raw_truth.py`；`old/channel_metadata.py` 继续生成生产通道元数据。
它们不是验收标准副本，不能因删除验收声明而删除，也不作为独立验收真值的推导来源。

## YAML 加载与版本绑定

- `dataset_id` 是配置短标识；`expected_root_dataset_id` 是待检根属性的长标识，
  与 `product.id/version/variant`、`template_version`、根 `schema_version`、
  内容 `dataset_version` 和 `data_revision` 分别保存，不混用版本含义。
- 严格拒绝重复键、未知字段、YAML alias/merge、未使用变量和不支持的布局/运算。
  目前支持完整帧分块、float16/float32 主数据、6/24 小时时间轴及现有规范化数组角色。
- 预处理直接读取声明，支持 zscore、multiply→zscore、divide→zscore、
  hypot→zscore、multiply→clip_min→log1p。公式只检查已支持的明确形式，不执行任意表达式。
- NPY 网格归属通过 `grid_provenance` 显式绑定，缺失时不能把形状相同当作坐标已获证明。
- 相对数据、附件及固定有效区域证明路径相对所属 YAML 解析；执行 JSON 中的 schema、数据及回执
  路径相对该 JSON 解析，不依赖执行时工作目录。
- 报告及转换回执绑定配置标识、模板版本、产品版本和 YAML 原始字节 SHA256；
  旧 profile 报告不自动升级为新证据。
  加载后源 YAML 变化会阻断本次通过；`status: draft` 不具备已批准标准的放行资格。

### 标准与执行证据分离

YAML 中不接受 `snapshot_id`、`production_evidence`、`conversion_evidence`。
这些运行绑定由单独 JSON 的 `execution` 对象提供，
不是另一套产品标准。否则“回执绑定 YAML 摘要、YAML 又记录回执摘要”会循环失效。
固定 YAML、生成回执、将回执引用写入执行 JSON，全程无需回写 YAML。

例如在 `run.json` 中填写：

```json
{
  "schema": "/path/to/era5_land_c8_1p5_h24.yaml",
  "dataset_path": "/path/to/frozen-dataset.zarr",
  "execution": {
    "snapshot_id": "production-snapshot-id",
    "production_evidence": {
      "path": "/path/to/frozen-production-record.json",
      "sha256": "<该记录原始字节的64位SHA256>"
    }
  }
}
```

需要转换证明时，在 `execution` 中增加 `conversion_evidence`，格式同上；
`conversion_required` 只能加强、不能关闭 YAML 已要求的门槛。单套执行配置的 `schema`
和转换专项配置的 `target_schema` 旁同样允许 `execution`。相对路径按该 JSON 所在目录解析。
单套流程读取回执前验证 SHA256，再调用 `evidence.py` 核验身份、必需检查覆盖和结果，
并记录到本次验收报告；不再使用独立 evidence 模块，不重新计算源/目标数值。
解析和 SHA256 使用同一份原始字节，运行前后均检查执行 JSON 未变化；
报告保存其路径和摘要，并继续独立记录 YAML 摘要。

## 当前验收范围

| 分类 | 已纳入检查 |
|---|---|
| 身份与声明 | 根 ID、版本、修订号、目录映射、规范化属性、配置摘要 |
| 原始布局 | 数组名、维度顺序、shape、dtype、chunks、编码、合并元数据一致性 |
| 时间 | 完整时间轴逐值对照，UTC、无 NaT/重复/缺点/多点/乱序，频率和相位 |
| 通道 | 完整有序清单、逐通道字典、已确认层次及处理声明 |
| 网格 | 全部经纬度坐标；CRA c71 南到北，其余北到南 |
| 统计量 | 主 Zarr mean/std、float32、有限/std>0、缩放字段一致、tp 0/1 |
| 内嵌掩膜 | 产品 MASK，mask/mask_channel、二值、标签顺序、逐像素互补 |
| 辅助结构 | 已登记 const/weight、原始布局、坐标、有限值；不限定 NC 变量名称 |
| 权重 | weight>=−1e-7，至少一个正值；容差内负值警告，超出失败 |
| 主块原值 | 严格解码、原始 shape/dtype、无 Inf、有限 tp>=0 |
| 存储证据 | 小数组全部物理键、主数组所选块键、扫描前后摘要及块指纹 |
| 完成与报告 | 已选块全部完成、读失败和缺失计数、JSON/逐块证据 |

NaN 比例、有限值范围和常数场只保存诊断，不设验收阈值。ORAS5 盐度的
units/source_units 接受值尚未定稿，本版不判定这两个字段；其余已确认结构和声明仍检查。
外部 mask 文件不打开、不参与验收。未登记 NC/NPY 为警告，不能用它们补齐主数据。
这些范围外项目不会以 needs-review 阻断本版 quick，也不会被标记为已通过。

本流程的默认 `strict-standard` 范围仍保留显式 QualityRule、生产/转换证据能力；
它不因十份 YAML 使用 confirmed-format 而被整体放宽。confirmed-format 不接受
QualityRule；新规则经确认后需显式建立新的配置/标准版本再纳入。

## 如何执行

直接调用公共 Python 接口：

```python
from pipelines.dataset import validate_zarr

# 只返回报告，不保存文件。
report = validate_zarr("/data/land.zarr", "config/dataset_specs/era5_land_c8_1p5_h24.yaml")

# 指定文件路径时保存 JSON。
report = validate_zarr(
    "/data/land.zarr",
    "config/dataset_specs/era5_land_c8_1p5_h24.yaml",
    output="/data/reports/land.json",
)
```

有独立生产或专项证明时，外围使用 `load_validation_run` 装配执行 JSON：

```python
from pipelines.dataset.configuration import load_validation_run

spec, locations = load_validation_run("/path/to/run.json")
report = validate_zarr(locations["dataset_path"], spec, output="/path/reports/land.json")
```

输出必须在输入数据和附件目录之外，也不能覆盖 YAML、执行 JSON 或输入回执。
多套验收由外围逐个调用；不提供目录选择或内部批量运行器。

## quick 与 full 的结论

quick 完整检查元数据、坐标、统计量、内嵌掩膜和已登记辅助结构，主数据默认读首、中、末
三个预期块，同时查物理块键。quick_passed=true 要求这些已选块全部完成且所有适用
quick 条款通过。max-blocks 导致抽样未读完、中断、读取失败、缺块或格式错误均不能通过。

full 读取全部独立预期块、查全部物理覆盖和意外块键，并要求独立生产记录证明数据已完成
生产且停止写入/快照不可变。缺少生产记录、SHA256 不匹配或物理覆盖不支持，不能宣称
full 完成。正式使用默认 full 且不传 max_blocks；是否保存报告由 output 决定。
生产记录字段及绑定规则详见计划 J06。代码可生成部分诊断报告，不能替生产端签署完成证明。

quick 报告的 scan_complete=false、release_passed=false 是模式定义，不能据此判定 quick
失败。生产冻结证明在 quick 中独立记录为未执行；在 full 中是必要条件。
release_passed 仅代表所写 validation_policy 范围的验收，不证明范围外项目或所有模型兼容。

## 查看报告

指定 output 时生成一个 JSON，包含完整 check_id、状态、expected/observed/evidence、coverage/quality
及 quick/full 完成结论。Python 返回的报告对象与保存内容一致；保存后的路径位于
`report.identity["report_path"]`。逐块记录可由外围通过 `on_block` 回调自行保存。

pass=已执行通过，fail=违反已纳入规则，warning=非阻断问题，not-applicable=范围外，
not-run/blocked=未执行或不能执行。quick_blocking_checks 表示 quick 阻断条目；
blocking_checks 也包括 full 尚未执行条件，不要混用。调用方从报告对象读取检查状态和完成结论；配置或输出错误通过异常返回。历史报告不会被重新解释或修改。

## 程序测试

```bash
python -m pytest -q
python -m pytest tests/dataset_building/acceptance -q
```

Windows 可显式指定项目内唯一 `--basetemp=.test-tmp/test-YYYYMMDD-HHMMSS`。
服务器指定 `$PWD/scratch/test-YYYYMMDD-HHMMSS`。测试使用小型真实 Zarr 与独立故障
注入，验证不规范格式、时间轴、通道、掩膜、权重、删除/损坏块、填充值掩盖、分片支持边界、
扫描中断及确认范围的失败边界。未配置外部旧格式集成样本时，相关集成测试可显示 skipped。
程序回归通过与服务器数据通过是两个独立结果，需分别查看。
执行配置回归覆盖不改 YAML 接入真实转换回执、摘要不匹配、运行中变更和输出保护；
分层回归在独立进程验证 datakit 不导入 pipelines、数据入口不导入外部模型依赖、生产和验收定义互不加载、
旧数据集验收入口已移除、数据集验收使用 Python API。
回归明确拒绝已删除的模型配置字段，并验证三个格式转换/刷新示例和原始真值提取仍可导入。


## 内存预算与进程历史峰值

`validate_zarr(..., memory_budget_mib=2048)` 在开始前检查预计工作内存和
当前进程驻留内存（RSS），扫描期间按块采样当前 RSS，超过预算时中断并保留不完整报告。
这属于协作式检查，不是操作系统强制内存上限；块间采样不能保证捕获每个瞬时峰值。

进程之前执行其他任务时产生的历史峰值不会阻止新的扫描。报告中的字段含义为：

- `process_resident_before_bytes`：当前校验开始前的驻留内存。
- `sampled_peak_resident_bytes`：从本次校验开始采样到的最大驻留内存。
- `peak_process_memory_bytes`：进程自启动以来的历史峰值，仅供诊断，不参与预算判断。

Linux 从 `/proc/self/statm` 读取当前 RSS，Windows 读取当前 working set；其他系统可在
安装 psutil 时读取。无法测量时记录为 `null`，仍执行预计内存预算检查。
当前 RSS 确实超过预算时仍会阻止扫描，不因历史峰值处理的修复而放行。
