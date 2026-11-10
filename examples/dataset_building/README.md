# 标准数据集构建

本目录提供可直接运行和调试的构建、追加与校验示例，自定义加工写在 `example_build_*.py` 中，通用流程由 `pipelines.dataset` 执行：

```text
加载契约与源数据 → process → 标准元数据组装 → 产物检查与写出 → 落盘 quick 校验
```

## 运行构建用例

Fengshun 的 c76 / 1.5° 日统计 NC 衔接、配置及独立构建 cron 见
[Fengshun 每日 Zarr 构建手册](FENGSHUN_DAILY_OPERATIONS.md)。
入口为 `example_build_fengshun_daily.py`（单日）及 `build_ready_fengshun_days.py`（自动消费水位），
复用已有 S2S c76 标准和 Tongyun 发布流程，日均/日累计不再做小时求和。

Tongyun服务器的完整流程、当前crontab、日期水位、失败恢复及排查命令见
[每日下载与Zarr自动构建运行手册](TONGYUN_DAILY_OPERATIONS.md)。

### Tongyun 每日下载 → 日 Zarr

`example_build_tongyun_daily.py` 是 `example_build_era5.py` 的按日入口，沿用现有标准加工：
六小时求和、单位转换、降水 log1p、固定统计量 z-score 和标准 c78 产品契约。
本入口不增加 Tongyun 专用的单位换算或填值逻辑。输出是归一化 Zarr，不是物理域模型输入。

每日源声明 `config/data_sources/era5_cds_daily_raw.yaml` 匹配下载器的 `raw/daily` 根目录：

```text
pl_h6/<variable>/<year>/<variable>_YYYYMMDD.nc
sfc_inst_h6/<variable>/<year>/<variable>_YYYYMMDD.nc
sfc_accum_h1/<variable>/<year>/<variable>_YYYYMMDD.nc
```

每次只通过临时符号链接暴露目标日的18个文件和前一天的5个累计量文件，
不扫描全部历史、不复制大文件、不修改原文件。精确模板排除 `.part`。
目标日00时需要前一天19–23时的累计量；即使其他文件已齐，也不能省略这5份边界文件。
临时目录在结束时清理，原文件保留；构建报告中的源检查路径是临时链接目录。

在服务器仓库根目录运行（`/path/to` 附件路径必须换成实际路径）：

```bash
python -m examples.dataset_building.example_build_tongyun_daily \
  --date 2026-10-30 \
  --source /home/xmetai/era5_download/era5_tongyun_c78/raw/daily \
  --destination /home/xmetai/era5_download/era5_tongyun_c78/zarr/era5.20261030.c78.p25.h6.v1 \
  --mean-nc /path/to/mean.nc \
  --std-nc /path/to/std.nc \
  --mask-nc /path/to/mask.nc \
  --const-nc /path/to/const.nc \
  --weight-npy /path/to/weight.npy \
  --report-dir /home/xmetai/era5_download/era5_tongyun_c78/reports \
  --dry-run
```

先用 `--help` 检查参数；dry-run 会执行数据加工和检查并保存报告，不写 Zarr。
确认通过并决定创建上述新输出后，将 `--dry-run` 替换为 `--write`。
输出必须尚不存在，重复执行不会覆盖或追加。每日日 Zarr 是独立产品，包含
00/06/12/18 UTC共4帧，`data(time, channel, lat, lon)`，全球网格为 `(4,78,721,1440)`。

mean/std 必须使用与现有加工域匹配的固定78通道统计量，按日入口不从当天数据重新计算统计量。
mask、const、weight 使用标准契约匹配的外部附件。NaN等质量问题按现有构建器规则报告，
本入口不会自动填补。先前下载首日若没有前一天累计量，可从下一个下载完整日开始构建。

定时编排时分别跟踪下载日期和构建成功日期：对每个下载完成且尚未构建的日期调用本入口，
只有退出码为0、报告显示写入及Quick校验通过才交给推理。不要只检测目标目录存在，
因为构建失败也可能留下未通过验收的产物。这里未修改下载cron，也未自动启动推理或发布。

### 下载完成后自动构建

`build_ready_tongyun_days.py` 读取原下载器的 `watermark.json`，与下载进程独立运行。
`latest_complete_day` 是允许构建的最晚日期；水位不存在时输出 WAIT，不开始构建。
从本地温度日文件发现候选日期，只处理水位及之前的日期，并核对当天18个文件及前一天
5个累计量文件全部存在。仍在下载的 `.part` 不参与候选日发现或文件齐全检查。
不再读取 `status.jsonl`；正式的时间、变量和质量验收仍由构建器执行。
水位推进多天时，按日期顺序补建尚未生成的日产品，不会只保留最新一天。
每次仍扫描温度文件名以发现候选日期，但先检查完成标记和最终Zarr目录；
两者都存在就直接跳过，不再检查该日期的23份源文件。只有尚未完成的日期才检查源文件。
无需修改下载器，也不等待下载器把全部追赶日期下载完。

先修改 `examples/dataset_building/tongyun_daily_server.yaml` 中五个附件路径，
source/download_watermark 必须对应现有下载器的数据目录与水位文件。
旧配置需将 download_status 替换为 download_watermark，并把路径改为 watermark.json。
运行 dry-run（会读数据、加工和保存诊断，但不创建Zarr或完成标记）：

```bash
python -m examples.dataset_building.build_ready_tongyun_days \
  --config examples/dataset_building/tongyun_daily_server.yaml --dry-run
```

确认后使用相同命令的 `--write` 替换 `--dry-run` 创建新产物。
默认每次最多处理一个就绪日期；使用 `--max-days 3` 可单次追赶三个日期。
mean/std 固定使用已有统计量，转换处理仍来自 `example_build_era5.py`。

现有下载cron保持每天10:30，另加一条每30分钟检查构建的cron。构建代码所在服务器仓库是
`/home/xmetai/xmetai-datakit`，下载器仍在 `/home/xmetai/xmetai-datakit-download`：

```bash
mkdir -p /home/xmetai/logs/xmetai-era5-tongyun
crontab -e
```

新增完整一行：

```cron
*/30 * * * * cd /home/xmetai/xmetai-datakit && /usr/bin/flock -n /tmp/xmetai-era5-tongyun-zarr.lock /home/xmetai/anaconda3/envs/zjf/bin/python -u -m examples.dataset_building.build_ready_tongyun_days --config examples/dataset_building/tongyun_daily_server.yaml --write >> /home/xmetai/logs/xmetai-era5-tongyun/zarr-build.log 2>&1
```

构建锁与下载锁不同，可以一边下载下一天一边构建上一天；同一时刻只有一个构建任务。
新一天满足就绪条件后在下一次半小时检查时尝试启动，若上一次构建未结束则等待后续触发。

构建先写到 `output_root/_attempts/<随机ID>/<产品名>`，只有构建器返回写入和Quick校验成功，
才将产物移到 `output_root/<产品名>`，再原子写 `_completed/YYYYMMDD.json`。
后续检查会跳过同时有成功标记和最终目录的日期。dry-run和失败都不标记完成。
失败产物保留在 `_attempts` 供检查，下次使用新目录重试，不覆盖旧数据；需监控磁盘占用。
如果进程恰在发布目录与写标记之间退出，消费者会报告“目录存在但没有完成标记”，
拒绝覆盖，需要检查报告与产物后人工恢复。原报告的输出路径指向对应attempt，
完成标记的destination记录最终发布路径。

```bash
tail -f /home/xmetai/logs/xmetai-era5-tongyun/zarr-build.log
ls /home/xmetai/era5_download/era5_tongyun_c78/zarr/_completed/
```

日志中的 `WAIT` 表示水位尚不存在或源文件未齐全，`BUILD` 表示开始加工，
`READY` 表示日Zarr已发布，`SKIP` 表示该日期已成功生成，`FAILED` 查看JSON报告。
此衔接仅自动构建Zarr，尚不自动调用模型推理。

### 通用构建命令

在仓库根目录通过命令行传入契约、数据和附件路径，无需修改脚本。
以下命令测试当前下载的 2025 年 1 月数据：

```bash
python -m examples.dataset_building.example_build_oras5 \
  --source-config config/data_sources/oras5_cmems_daily_raw.yaml \
  --target-spec config/dataset_specs/oras5_c88_p25_h24.yaml \
  --start-time 2025-01-01T00:00 --end-time 2025-01-31T00:00 \
  --dataset-id oras5.202501.c88.p25.h24.v2 --content-version v2 --data-revision 20261104 \
  --source /home/zjf/xmetai-datakit-data/downloads/oras5_daily \
  --destination /home/zjf/xmetai-datakit-data/zarr/pipeline_with_aux/oras5.202501.c88.p25.h24.v2 \
  --mean-nc /home/zjf/xmetai-datakit-data/oras5/oras5_process/stats/0p25/mean_88_19979.nc \
  --std-nc /home/zjf/xmetai-datakit-data/oras5/oras5_process/stats/0p25/std_88_19979.nc \
  --mask-nc /home/zjf/xmetai-datakit-data/oras5/oras5_process/stats/0p25/mask.nc \
  --const-nc /home/zjf/xmetai-datakit-data/static/0p25/const.nc \
  --weight-nc /home/zjf/xmetai-datakit-data/static/0p25/weight.nc \
  --report-dir /home/zjf/xmetai-datakit-data/reports \
  --dry-run

python -m examples.dataset_building.example_build_era5 \
  --source-config config/data_sources/era5_cds_monthly_raw.yaml \
  --target-spec config/dataset_specs/era5_c78_p25_h6.yaml \
  --start-time 2025-01-01T06:00 --end-time 2025-01-31T18:00 \
  --dataset-id era5.202501.c78.p25.h6.v1 --content-version v1 --data-revision 20261104 \
  --source /home/zjf/xmetai-datakit-data/downloads/era5_download \
  --destination /home/zjf/xmetai-datakit-data/zarr/pipeline_test/era5.202501.c78.p25.h6.v1 \
  --mean-nc /home/zjf/xmetai-datakit-data/stats/mean.nc \
  --std-nc /home/zjf/xmetai-datakit-data/stats/std.nc \
  --mask-nc /home/zjf/xmetai-datakit-data/stats/mask.nc \
  --report-dir /home/zjf/xmetai-datakit-data/reports \
  --source-stage raw --accumulation-hours 6 \
  --dry-run
```

两个构建示例默认 dry-run，执行加工与内存产物检查，不写出、不执行落盘校验。
`--dry-run` 可显式指定此行为；正式构建将其替换为 `--write`，二者互斥，输出必须不存在。
mean/std 参数可选，缺失时计算；mask 参数必填。
`--report-dir` 必填，完整构建报告保存到指定目录。
文件名包含数据集目录名、UTC 时间戳和随机标识，采用缩进 JSON，不覆盖已有报告。
两个入口结束时向终端打印简短摘要：运行结果、数据集、输出路径、写入与附件复制状态、
符合性/完整性/质量结论、Quick 校验、完整发布验收及报告路径。实际失败和未执行的
发布验收项目分开显示，逐通道统计等完整内容仅写入 JSON。
dry-run 和构建失败也会保存诊断。退出码 0 表示通过，1 表示构建失败。

直接复用 `config/dataset_specs` 中的标准契约，不再维护 `contracts/` 月份副本。
标准保留产品通道、网格、步长、类型、附件和数值规则；本次覆盖范围与身份通过
`--start-time/--end-time/--dataset-id` 指定。时间范围包含两端，必须符合标准 UTC 步长；
覆盖时间时必须显式指定新数据集标识，建议与目标目录名一致。
内容版本和修订号可用 `--content-version/--data-revision` 指定，省略时沿用标准值。
未提供覆盖参数时严格使用原标准中的完整历史范围和身份。

上述 ERA5 任务覆盖 2025-01-01 06:00 至 2025-01-31 18:00，共 123 个时次；
ORAS5 为一月 31 个日时次。标准 YAML 不被改写，完整报告记录标准路径、摘要、
运行覆盖参数及实际生效的时间范围、身份和版本。

ERA5 通过显式 `read=read_source` 逐变量读取 NC/ZIP 分区；读取阶段用
`time_<变量>` 保留各变量原始时间轴，不进行累积求和或通道打包。
在 `process` 中，瞬时量选目标时刻，`tp/ssr/ssrd/fdir/ttr` 将此前六个小时的量
（T-5h 至 T）求和，再合并、打包。缺小时会报错，NaN 不跳过；月初 00 时
需要前一月末数据。命令行默认 `--source-stage raw`；已打包且已完成累积的物理量
使用 `--source-stage packed` 并提供匹配的 packed 源契约，避免重复求和。
`--accumulation-hours` 默认 6，同时用于窗口求和及辐射单位换算，须与目标契约和输入语义一致。

ORAS5 的 `process` 按源映射读取原变量，按目标契约声明的物理深度选层，
转换经度到 0–360 并精确对齐目标网格，缺失纬度填 NaN，再按目标顺序打包。
目标通道名由所选标准决定，深度不使用固定索引。两个 ORAS5 标准统一使用
`so/to/uo/vo` 与 `19979`，分别对应原变量 `so_oras/thetao_oras/uo_oras/vo_oras`；
19979 标签的物理深度为 199.7900 m。原始数据源契约负责上述变量映射，
mean/std 直接按修订后的通道名选择，不兼容旧名称。
两者的 `process` 显式调用 `pipelines.dataset.preparation.pack_source_channels`，
复用坐标对齐、物理层选择和通道打包。ERA5 累积量处理保留在 ERA5 示例，
公共 pipeline 不包含数据集专属算法。

两个 `process` 也共用 `preparation` 中的准备工具：

- `load_external_array`：读取一个外部数组，加载后关闭文件。
- `prepare_statistics`：加载 mean/std，在缺失时基于当前加工域计算，选择目标通道并转 float32。
- `prepare_standard_arrays`：检查原始 mask、精确对齐数组、转换目标类型与坐标、设置分块并返回 Dataset。

这些工具由加工函数显式调用。ERA5 在单位换算和降水 log1p 后准备统计量，
ORAS5 在原物理量域准备统计量；归一化仍由各自的 `process` 调用 `apply_zscore`。

外部 `const.nc`、`weight.nc/weight.npy` 的要求由目标契约控制。
ORAS5 可传 `--const-nc/--weight-nc`，ERA5 可传 `--const-nc/--weight-npy`；
也可重复指定 `--auxiliary 文件名=路径`，文件名必须在目标契约中声明且不能重复。
正式写出必须提供全部必需附件，写入前只核对文件名、路径、存在性和 SHA256，
不检查附件内容。主数据写出后复制附件到目标根目录，再执行 Quick 内容校验；
不覆盖已有文件，不改变源文件，源文件变化或复制摘要不一致会失败。
dry-run 可省略附件，此时只检查主数据产物，报告记录缺少的附件；提供附件时
检查路径及摘要但不复制、不做内容校验。ERA5 的 weight.npy 须另行准备，
不会把 weight.nc 自动转换为 NPY。
已有 Zarr 仍不能作为构建输出；补附件后重测需单独操作，或用尚不存在的新目录完整构建。

| 用例 | 目标规格 | 专属加工 |
| --- | --- | --- |
| `example_build_era5.py` | `era5_c78_p25_h6` | 小时量聚合、压力层打包、风速合成、单位换算、降水非负裁剪与 log1p、归一化 |
| `example_build_oras5.py` | `oras5_c88_p25_h24` | 原变量映射、物理深度选择、网格对齐、通道打包与归一化 |

两个用例均支持外部 mean/std；未提供时，要求源时间与目标完整一致，在归一化前按通道沿 time/lat/lon 计算（忽略 NaN，ddof=0）。统计量须符合目标契约。当前用例的 `mask_path` 必须提供。

## 开发自定义加工

在本目录新增 `example_build_<源数据>.py`，定义：

```python
def process(dataset, source_contract, spec, *, mean_path=None, std_path=None, mask_path=None):
    # 加工数据、准备附件，并设置目标 dtype、坐标和通道顺序。
    ...
    return product  # xarray.Dataset，包含 data、mean、std、mask
```

`process` 负责单位换算、派生变量、重采样、归一化及附件准备。使用 `datakit.io` 读取附件、`datakit.transforms` 组合基础算子；自行打开的附件应在关闭前加载所需数据。

pipeline 负责源资源管理、标准元数据组装、检查、写出和回读校验，不修复加工结果或自动执行目标契约中的预处理声明。返回数组的类型、布局、坐标和通道顺序须满足契约；统计量应对应归一化前的数据域。

| 基础算子 | 职责 |
| --- | --- |
| `compute_mean_std` | 按指定维度计算 mean/std，使用 float64 归约 |
| `apply_scale` | 按显式变量系数缩放 |
| `apply_log1p` | 对指定变量执行 log1p，不自动裁剪 |
| `apply_zscore` | 使用对齐的 mean/std 执行归一化 |

源契约参考 [`config/data_sources`](../../config/data_sources)，目标契约参考 [`config/dataset_specs`](../../config/dataset_specs)。选择与源单位、累计窗口及目标语义匹配的加工脚本；目标契约和输出路径须匹配本次产物的覆盖范围、网格、通道及身份。

## Python 任务接口

示例直接调用 `pipelines.dataset` 中的任务接口。需要在自己的程序中组织构建时，通过 `partial` 绑定加工参数：

```python
from functools import partial
from pipelines.dataset import build_zarr
from examples.dataset_building.example_build_era5 import process

report = build_zarr(
    source_config="/path/to/source.yaml",
    target_spec="/path/to/target.yaml",
    source_path="/data/input.nc",
    destination="/data/output/product.zarr",
    process=partial(process, mean_path="/data/mean.nc", std_path="/data/std.nc", mask_path="/data/mask.nc"),
    dry_run=False,
)
print(report.to_dict())
```

两个用例分别在自己的 `main(argv=None)` 中定义参数、调用构建、保存报告并返回退出码，
便于独立阅读和调整。公共构建、数据准备和报告写入能力仍使用 `pipelines.dataset`。
正常导入不执行任务，也可在 IDE 中配置命令行参数并调试。

## 追加与校验

修改 [example_append_era5.py](example_append_era5.py)、[example_validate_era5.py](example_validate_era5.py) 中的路径后，可直接运行或调试：

```bash
python -m examples.dataset_building.example_append_era5
python -m examples.dataset_building.example_validate_era5
```

也可直接调用：

```python
from pipelines.dataset import append_zarr, validate_zarr

# 已有 B + 后续 A → 独立 C，内部执行完整校验。
report = append_zarr(
    source="/data/A.zarr", target="/data/B.zarr", destination="/data/C.zarr",
    spec="/path/to/combined_target.yaml", output="/data/reports/append.yaml",
)
print(report.release_passed)

report = validate_zarr(
    target="/data/C.zarr", spec="/path/to/combined_target.yaml",
    mode="full", output="/data/reports/validation.yaml",
)
print(report.release_passed)
```

追加按 B + A 生成独立 C，保留输入且拒绝已有输出，写出后内部执行 full 校验。契约须描述完整 BA，并准备相应外部附件及发布证据。追加期间输入保持静止，预留完整 BA 的额外磁盘空间；内存至少容纳一个时间片及静态附件。

独立校验只检查目标产物与契约及声明规则，不执行源到目标的加工对照；默认 full，按 `release_passed` 判断通过；quick 按 `quick_passed` 判断，不代替完整验收。`output=None` 不保存报告；指定校验报告时仅接受新的 `.yaml` 路径。

接口参数、模块职责、资源管理及失败处理统一见 [数据集任务接口](../../pipelines/dataset/README.md)。
