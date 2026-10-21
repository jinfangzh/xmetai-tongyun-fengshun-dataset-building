# 标准数据集校验

当前规范见 [标准数据集校验计划](validation/VALIDATION_PLAN.md)。计划 1.4、十份实例
profile 1.6 使用 `confirmed-format` 范围，仅验收已确认规则。主数据输入始终只读。
流程内产品定义冻结每个产品的完整通道顺序、网格、已确认处理声明、内嵌掩膜和辅助结构；
profile 冻结每批的时间、路径和修订号。其他批次应新建 profile，不反推期望值。

2026-10-21 调整代码归属：产品定义和专用校验集中在 `pipelines/dataset_building/`，
配置集中在 `examples/dataset_building/configs/`。公共 schema 暂不新增产品标准，后续由
维护者提炼共性并设计 YAML 声明。此次目录迁移不改变计划 1.4 和配置 1.6 的规则。

## 模块职责与流程入口

| 位置 | 职责 |
|---|---|
| `pipelines/dataset_building/definitions/standard_dataset.py` | 共用产品、网格、通道、掩膜、实例和证据类型 |
| `pipelines/dataset_building/definitions/*_c*.py`、`dataset_products.py` | 按产品独立声明标准，显式选择产品版本 |
| `pipelines/dataset_building/definitions/era5_c116.py`、`channel_attributes.py` | c116 生产常量、来源映射及生产通道元数据生成；转换和提取流程使用新导入 |
| `datakit/io/_zarr_inspection.py` | 严格只读元数据、原始块及物理键定位 |
| `pipelines/dataset_building/validation/standard_dataset.py`、`_standard_auxiliary.py`、`_standard_quality.py` | 已打开对象的规则检查和有界逐通道统计，不负责路径打开或报告写入 |
| `pipelines/dataset_building/validation/reports.py` | 统一状态、quick/full 结论和 JSON 安全转换 |
| `pipelines/dataset_building/_standard_validation.py` | 输入预检、辅助对象管理、证据绑定和最终覆盖结论 |
| `pipelines/dataset_building/_standard_scan.py`、`_validation_runtime.py` | 主块扫描、物理覆盖、进度、文件变化及进程内存测量 |
| `pipelines/dataset_building/config.py`、`reporting.py` | 配置加载、报告落盘 |
| `examples/dataset_building/` | CLI 入口和 configs 配置；独立 core 探测不引入核心库依赖 |
| `tests/validation/` | 按扫描、布局、产品、质量、掩膜、辅助、确认范围、对照、CLI 和报告分组的回归 |

```python
from pipelines.dataset_building import validate_standard_dataset
from pipelines.dataset_building.definitions import DatasetValidationSpec, get_dataset_product
from pipelines.dataset_building.validation.reports import ValidationReport, to_jsonable
```

流程入口接收结构化输入并返回报告对象；config/reporting 负责配置加载和报告保存，示例负责 CLI 参数。
产品模块共享共用类型和辅助定义，每个产品仍独立维护通道、网格和 MASK。
转换/组合专项与单套校验共享证据和报告类型，保留各自入口。

所有产品模块采用相同的具名声明：

| 声明 | 类型 | 含义 |
|---|---|---|
| `CHANNELS` | `tuple[str, ...]` | 默认产品版本的完整有序通道名 |
| `MASK` | `MaskDefinition` | 产品的内嵌掩膜标准 |
| `VARIANTS` | `tuple[DatasetVariant, ...]` | 网格、频率及辅助结构变体 |
| `PRODUCT` | `DatasetProduct` | 在本产品文件中组装的完整标准 |

`PRODUCT` 使用 `product_id/version/channels/variants/mask` 具名参数。
`dataset_products.py` 只汇总导入和选择，不构造个别产品。
ORAS5 另有显式 `CURRENT_CHANNELS/CURRENT_PRODUCT`，默认 `CHANNELS/PRODUCT` 对应旧版，
两个版本不自动转换。ERA5 c116 的 `CHANNELS` 与 `DYNAMIC_CHANNELS` 在流程内同一模块
维护，原网格、来源映射及生产元数据生成能力一并迁入，值及默认行为保留；公共 schema
不再导出这些产品接口。仓库外调用方需要更新导入路径。共用产品助手依赖通用维度及
类型，不依赖个别产品模块。生产元数据生成器不作为严格验收标准的推导来源。

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
| 完成与报告 | 已选块全部完成、读失败和缺失计数、JSON/CSV/Markdown/逐块证据 |

NaN 比例、有限值范围和常数场只保存诊断，不设验收阈值。ORAS5 盐度的
units/source_units 接受值尚未定稿，本版不判定这两个字段；其余已确认结构和声明仍检查。
外部 mask 文件不打开、不参与验收。未登记 NC/NPY 为警告，不能用它们补齐主数据。
这些范围外项目不会以 needs-review 阻断本版 quick，也不会被标记为已通过。

本流程的默认 `strict-standard` 范围仍保留显式 QualityRule、来源/消费者证据能力；
它不因十份 profile 使用 confirmed-format 而被整体放宽。confirmed-format 不接受
QualityRule；新规则经确认后需显式建立新的配置/标准版本再纳入。

## 如何执行

在仓库根目录运行。以下路径是历史部署环境，须先部署本次分支代码后才能使用新的 CLI 路径；
本次本地迁移未更新服务器。历史部署目录为
`/workspace/xujn/datakit-validation-20261012-4993ba64`，解释器为
`/workspace/xujn/Data_Format_Refresh/.runtime/bin/python`。离线测试依赖在部署目录的
test-dependencies 中，无须修改共享运行环境。

```bash
cd /workspace/xujn/datakit-validation-20261012-4993ba64
export PYTHONPATH="$PWD/test-dependencies"
export TMPDIR="$PWD/scratch"
export TEMP="$PWD/scratch"
export TMP="$PWD/scratch"
PY=/workspace/xujn/Data_Format_Refresh/.runtime/bin/python

"$PY" -m examples.dataset_building.validate_standard_dataset \
  --inventory examples/dataset_building/configs/profiles/inventory.json \
  --mode quick --sample-blocks 3 \
  --output /workspace/xujn/validation-reports/quick-YYYYMMDD-HHMMSS
```

`--output` 指定报告目录，必须在全部输入目录之外；测试、报告和临时运行文件只写入
/workspace/xujn。更换数据根目录用 `--data-root`；单套用 `--profile` 和可选
`--dataset-path`，不要将单套路径参数与 inventory 混用。

## quick 与 full 的结论

quick 完整检查元数据、坐标、统计量、内嵌掩膜和已登记辅助结构，主数据默认读首、中、末
三个预期块，同时查物理块键。quick_passed=true 要求这些已选块全部完成且所有适用
quick 条款通过。max-blocks 导致抽样未读完、中断、读取失败、缺块或格式错误均不能通过。

full 读取全部独立预期块、查全部物理覆盖和意外块键，并要求独立生产记录证明数据已完成
生产且停止写入/快照不可变。缺少生产记录、SHA256 不匹配或物理覆盖不支持，不能宣称
full 完成。正式使用 `--mode full` 且不传 `--max-blocks`；输出仍指定 /workspace/xujn。
生产记录字段及绑定规则详见计划 J06。代码可生成部分诊断报告，不能替生产端签署完成证明。

quick 报告的 scan_complete=false、release_passed=false 是模式定义，不能据此判定 quick
失败。生产冻结证明在 quick 中独立记录为未执行；在 full 中是必要条件。
release_passed 仅代表所写 validation_policy 范围的验收，不证明范围外项目或所有模型兼容。

## 查看报告

- summary.md：十套结论及警告、未执行条目，可直接阅读。
- summary.csv：重点看 quick_passed、quick_scan_complete、quick_blocking_count、validation_scope。
- 每套 JSON：完整 check_id、状态、expected/observed/evidence 和 coverage/quality。
- blocks.jsonl：每个已选逻辑块、物理键、读取状态及原值异常。
- batch.json：请求数量、完成数量、是否中断和批次结束时间。

pass=已执行通过，fail=违反已纳入规则，warning=非阻断问题，not-applicable=范围外，
not-run/blocked=未执行或不能执行。quick_blocking_checks 表示 quick 阻断条目；
blocking_checks 也包括 full 尚未执行条件，不要混用。CLI 退出码 0/1/2/3 分别代表
本次范围通过/规则失败/未完成或证据不足/配置输出错误。历史报告不会被重新解释或修改。

## 程序测试

```bash
python -m pytest -q
python -m pytest tests/validation -q
```

Windows 可显式指定项目内唯一 `--basetemp=.test-tmp/test-YYYYMMDD-HHMMSS`。
服务器指定 `$PWD/scratch/test-YYYYMMDD-HHMMSS`。测试使用小型真实 Zarr 与独立故障
注入，验证不规范格式、时间轴、通道、掩膜、权重、删除/损坏块、填充值掩盖、分片支持边界、
扫描中断及确认范围的失败边界。未配置外部旧格式集成样本时，相关集成测试可显示 skipped。
程序回归通过与服务器数据通过是两个独立结果，需分别查看。


## 内存预算与进程历史峰值

`validate_standard_dataset(..., memory_budget_mib=2048)` 在开始前检查预计工作内存和
当前进程驻留内存（RSS），扫描期间按块采样当前 RSS，超过预算时中断并保留不完整报告。
这属于协作式检查，不是操作系统强制内存上限；块间采样不能保证捕获每个瞬时峰值。

进程之前执行其他任务时产生的历史峰值不会阻止新的扫描。报告中的字段含义为：

- `process_resident_before_bytes`：当前校验开始前的驻留内存。
- `sampled_peak_resident_bytes`：从本次校验开始采样到的最大驻留内存。
- `peak_process_memory_bytes`：进程自启动以来的历史峰值，仅供诊断，不参与预算判断。

Linux 从 `/proc/self/statm` 读取当前 RSS，Windows 读取当前 working set；其他系统可在
安装 psutil 时读取。无法测量时记录为 `null`，仍执行预计内存预算检查。
当前 RSS 确实超过预算时仍会阻止扫描，不因历史峰值处理的修复而放行。
