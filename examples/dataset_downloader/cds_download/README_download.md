# CDS 月度下载工具

该示例调用 CDS 下载原始文件或服务端日统计结果，不做本地日统计、解压或 Zarr 转换。

## Profile：按物理分组

小时配置保留 5 组：气压层、大气单层（近地面、云与辐射共 21 变量）、ERA5-Land 土壤、海浪、静态场。

日统计设计分为 3 组，预期的合并 Profile 为：

| Profile | 统计内容 | 单月请求数 |
|---|---|---:|
| `derived-era5-single-levels-daily-statistics_sfc_v1.yaml` | 23 个单层变量日均、5 个变量日累计 | 28 |
| `derived-era5-single-levels-daily-statistics_t2m_v1.yaml` | 温度日均、日最大、日最小 | 3 |
| `derived-era5-pressure-levels-daily-statistics_pl_v1.yaml` | 5 个变量 × 13 层的日均 | 5 |

当前目录尚未包含上表的三个 `*_v1.yaml` 合并 Profile，`batch_daily_task.yaml`
引用了这些缺失文件，暂时不能直接运行。现有的日统计 Profile 是
`sfc_mean`、`sfc_sum`、`t2m_max`、`t2m_min` 和 `pl_mean` 等独立文件；
需先调整 Batch 配置或补齐合并 Profile。下面的日统计结构与输出说明记录的是预期方案。

气压层在同一个变量请求中一起下载，因此单月共 36 请求。温度日均已从 sfc 移到 t2m，不重复下载；土壤日均仍使用 ERA5 single levels 的四层数据。

```yaml
dataset: derived-era5-single-levels-daily-statistics
product_type: reanalysis
time_zone: utc+00:00
frequency: 1_hourly
area: null

statistics:
  daily_mean:
    variable:
      - 2m_temperature

  daily_max:
    variable:
      - maximum_2m_temperature_since_previous_post_processing

  daily_min:
    variable:
      - minimum_2m_temperature_since_previous_post_processing
```

`statistics` 是本地分组字段，下载器将每组展开成 CDS 原生的 `daily_statistic` 和 `variable`，不会把 statistics 传给服务端。共同的 frequency、time_zone、area 和 pressure_level 由各统计任务继承；统计子项只允许 variable。

最高/最低温变量保留旧配置的 since_previous_post_processing 定义，不等价于对每小时 2m_temperature 求极值；没有静默替换。日统计仅支持 ZIP 中的 NetCDF，不填写 time、data_format、download_format。frequency 的 1_hourly/3_hourly/6_hourly 是统计输入采样频率，输出仍是每日值。

小时 Profile 保持原生 variable/time/data_format/download_format 结构。旧式日统计 Profile（顶层 variable + daily_statistic）仍兼容；不要与 statistics 混用。旧式 Profile 保持原输出路径。

## Batch：运行设置

当前可直接运行的示例是小时任务 `batch_task.yaml`；`batch_daily_task.yaml`
用于描述日统计方案，但依赖上文提到的缺失 Profile。

```yaml
profiles:
  - path: ./profiles/derived-era5-single-levels-daily-statistics_t2m_v1.yaml
    output_subdir: t2m

storage:
  root_path: ./era5_download/daily

period:
  start_month: "2014-01"
  end_month: "2014-02"

execution:
  max_workers: 1
  max_retries: 3
  retry_delay_seconds: 30
  continue_on_error: true
```

月份手动填写，包含首末月；不支持自动月份选择。路径相对于 Batch 文件目录解析。每次请求为一个变量、一个统计方式、一个整月，自动按日历生成日期。

## 输出与记录

分组日统计输出为：

```text
era5_download/daily/
├── sfc/daily_mean/<variable>/<year>/<variable>_<YYYYMM>.zip
├── sfc/daily_sum/<variable>/<year>/<variable>_<YYYYMM>.zip
├── t2m/daily_mean/2m_temperature/<year>/2m_temperature_<YYYYMM>.zip
├── t2m/daily_max/<variable>/<year>/<variable>_<YYYYMM>.zip
├── t2m/daily_min/<variable>/<year>/<variable>_<YYYYMM>.zip
├── pl/daily_mean/<variable>/<year>/<variable>_<YYYYMM>.zip
└── _records/batch_daily_task/
    ├── plan.txt
    └── status.jsonl
```

不同统计结果使用独立目录，不覆盖。小时及旧式日统计的路径仍为 `<output_subdir>/<variable>/<year>/<variable>_<YYYYMM>.<suffix>`。同一 Batch 内重复目标路径会报错。

记录默认放在 `storage.root_path/_records/<Batch文件名不含扩展名>/`。plan.txt 保存本次计划，status.jsonl 追加记录；完成状态和目标文件同时存在才跳过下载。任务哈希包含请求和目标路径。原 records.plan_path/status_log_path、log_path 和 CLI 覆盖仍兼容；无 storage 的旧 Batch 须显式指定记录路径。

本次三组配置使用新的输出目录。旧文件及记录没有删除或自动搬移；旧完成记录不能直接匹配新路径，需要重新下载或单独核对迁移。后续修改统计口径、区域或网格时，应使用不同输出目录，避免跨 Batch 覆盖。仅存在文件但没有完成记录，不会自动认定下载成功。

下载先写 `.part`，完成后替换目标文件。失败按 max_retries 重试，等待 retry_delay_seconds；没有成功后的额外等待。文件内容的科学完整性未自动验证。

## 安装与执行

在仓库根目录安装独立依赖，并配置 CDS 凭据、接受相应数据集条款：

```bash
python -m pip install -r examples/dataset_downloader/cds_download/requirements.txt
```

先修改 `batch_task.yaml` 中的月份和输出目录，再运行小时任务：

```bash
# 预览请求，不连接 CDS；会生成本地计划文件
python examples/dataset_downloader/cds_download/cds_download.py examples/dataset_downloader/cds_download/batch_task.yaml --dry-run

# 只生成目标文件清单
python examples/dataset_downloader/cds_download/cds_download.py examples/dataset_downloader/cds_download/batch_task.yaml --plan-only

# 正式下载
python examples/dataset_downloader/cds_download/cds_download.py examples/dataset_downloader/cds_download/batch_task.yaml
```

`batch_daily_task.yaml` 目前不能直接运行，需先补齐或更换其 Profile 路径。
该目录没有 `tests/`，因此不再给出旧的 `unittest discover` 命令。

官方日统计字段及返回格式：[ERA5 日统计说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=462888259)。

## 数据集统一分类

`cds_download.py` 中的 `DATASET_SPECS` 集中登记数据集规则。Profile 仍只填写原生 dataset 名称，不重复声明分类。

| 数据集 | 请求类型 | 层次类别 | product_type 必填 | 输出后缀 |
|---|---|---|---|---|
| reanalysis-era5-single-levels | hourly | single | 是 | 按格式字段推导 |
| reanalysis-era5-pressure-levels | hourly | pressure | 是 | 按格式字段推导 |
| reanalysis-era5-land | hourly | land | 否 | 按格式字段推导 |
| derived-era5-single-levels-daily-statistics | daily | single | 是 | zip |
| derived-era5-pressure-levels-daily-statistics | daily | pressure | 是 | zip |
| derived-era5-land-daily-statistics | daily | land | 否 | zip |

前五种覆盖现有全部 Profile 使用的数据集，第六种保留已有接口支持。sfc、wave、static 等物理分组可共享同一个 single 数据集分类；不会因登记数据集而自动生成下载任务。

后续配置展开、字段校验和输出后缀均从分类表读取规则：daily 支持 statistics，pressure 要求非空 pressure_level，其他类别不接受 pressure_level。最终执行器继续使用生成好的 dataset/request/target，不再自行猜测数据集类别。

未登记的数据集会在构建任务时明确报错，不能默认为小时产品。支持新接口时，先在 DATASET_SPECS 登记并验证其规则，再添加 Profile。
