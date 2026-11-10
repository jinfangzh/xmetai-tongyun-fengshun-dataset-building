# Fengshun 每日下载 → c76 / 1.5° 日 Zarr

下载器 `run_fengshun_c76_daily.py` 保留 `raw/cds_daily_1p5` 中的16份日统计ZIP，
校验完成后更新下载水位。构建器直接读取这些下载产物，ZIP内的NC只在读取时临时解包，
不再要求永久保存一份 `raw/daily_1p5` NC副本。
本流程复用 Tongyun 的水位消费、独立日产品、失败重试和 Quick 校验后发布逻辑。
代码修改不代表已同步到服务器、安装构建 cron 或完成真实数据构建。

## 数据与加工

源契约是 `config/data_sources/era5_cds_fengshun_daily_raw.yaml`，源根目录必须是
`raw/cds_daily_1p5`，不再指向旧的 `raw/daily_1p5` 中间目录。

```text
cds_daily_1p5/
├── pl_mean/<variable>/<year>/<variable>_YYYYMMDD.zip  # z/t/u/v/q，13层日均
├── sfc_mean/<variable>/<year>/<variable>_YYYYMMDD.zip # 9个地面日均变量
└── sfc_sum/<variable>/<year>/<variable>_YYYYMMDD.zip  # tp/ttr 日累计
```

复用已有 `config/dataset_specs/s2s_c76_1p5_h24.yaml`，每日覆盖参数由入口指定，
不改写历史标准。产品名为 `s2s.YYYYMMDD.c76.1p5.h24.v1`，产品身份采用现有 S2S c76 契约。
每个产品只有当日 00 UTC 一个时间标签，`data(time,channel,lat,lon)` 形状为
`(1,76,121,240)`，float16；纬度90→−90，经度0→358.5。
通道顺序为 z/t/u/v/q，每个变量按1000→50 hPa排列，再接
t2m/d2m/sst/ttr/u10m/v10m/u100m/v100m/msl/tcwv/tp。

CDS 已计算 UTC 日均及日累计，本地不再进行小时累积，不需要前一天边界文件。
只通过临时只读符号链接暴露目标日16份ZIP；`.zip.part` 不计入完整输入。
共享NetCDF读取器临时解包，保持句柄直到加工完成，随后清理临时目录，原ZIP不修改。
先统一经度表达及经纬度顺序，再逐变量检查时间恰好是目标日00 UTC，网格与目标完全一致；拒绝缺格、错格及错时间，
不做插值、降采样或填值。之后打包通道并复用 `example_build_era5.process`：

- q：kg/kg ×1000 → g/kg。
- ttr：日累计 J/m² ÷86400 → W/m²。
- tp：m ×1000 → mm，裁剪非负后 log1p。
- 使用固定 mean/std 归一化；tp 的有效 mean=0、std=1，不重复 z-score。

必须准备匹配模型训练加工域的固定76通道 mean/std，不能使用 Tongyun c78 统计量，
也不能从当天数据临时计算。mask 为 `(2,121,240)`、uint8，通道 land_mask/sea_mask；
const.nc 为 `(2,121,240)`、float32，通道 z/lsm；weight.npy 为 `(1,121,1)`、float32。
附件必须匹配1.5°目标网格。SST 等 NaN 按现有契约报告，不自动修复。

## 配置与手动验证

### 从旧的NC中间目录方式升级

1. 同步下载器仓库 `run_fengshun_c76_daily.py`，改为校验ZIP后更新水位，不再生成永久NC。
2. 同步本构建仓库的Fengshun入口、源配置和 `fengshun_daily_server.yaml`；其中source
   必须改为 `/home/xmetai/era5_download/era5_fengshun_c76/raw/cds_daily_1p5`。
3. 不删除下载成功记录、水位或已有Zarr完成标记；旧NC副本保留但不再使用。
4. 如果ZIP已经全部下载、但旧流程在NC处理阶段失败导致没有水位，用新版下载器
   `--date YYYY-MM-DD` 重跑该日：跳过成功下载，校验ZIP并补写水位，不能手动伪造水位。
5. 先执行下方dry-run；有就绪数据并检查通过后再正式写出。没有就绪数据时WAIT不代表加工验收通过。

现有下载cron命令不用修改。构建cron入口也不变；只需保证使用更新后的构建仓库和配置。

修改 `examples/dataset_building/fengshun_daily_server.yaml`：下载根目录及水位采用下载器默认路径，
五个统计量和静态附件路径是部署模板，使用前必须替换或核实服务器上的真实路径。
输出和报告独立位于 `/home/xmetai/era5_download/era5_fengshun_c76/{zarr,reports}`。

以下假设构建仓库部署在 `/home/xmetai/xmetai-tongyun-fengshun-dataset-building`，Python 使用 xmetai 环境；
如果实际位置不同，配置、命令和 cron 中一并替换。在仓库根目录先检查：

```bash
cd /home/xmetai/xmetai-tongyun-fengshun-dataset-building
/home/xmetai/anaconda3/envs/xmetai/bin/python -m examples.dataset_building.build_ready_fengshun_days \
  --config examples/dataset_building/fengshun_daily_server.yaml --dry-run
```

默认每次处理最早的一个未完成日，`--max-days 3` 可追赶3天。
dry-run 会读取数据、加工及保存报告，不创建 Zarr 或完成标记。
确认输入、输出和报告后，将 `--dry-run` 替换为 `--write` 创建新的日产品。
输出必须不存在，源ZIP和外部附件不会被修改。

指定单日也可使用独立入口（示例日期按实际下载情况替换）：

```bash
/home/xmetai/anaconda3/envs/xmetai/bin/python -m examples.dataset_building.example_build_fengshun_daily \
  --date 2026-11-04 \
  --source /home/xmetai/era5_download/era5_fengshun_c76/raw/cds_daily_1p5 \
  --destination /home/xmetai/era5_download/era5_fengshun_c76/zarr/s2s.20261104.c76.1p5.h24.v1 \
  --mean-nc /home/xmetai/stats/fengshun_c76/mean.nc \
  --std-nc /home/xmetai/stats/fengshun_c76/std.nc \
  --mask-nc /home/xmetai/stats/fengshun_c76/mask.nc \
  --const-nc /home/xmetai/stats/fengshun_c76/const.nc \
  --weight-npy /home/xmetai/stats/fengshun_c76/weight.npy \
  --report-dir /home/xmetai/era5_download/era5_fengshun_c76/reports --dry-run
```

单日入口不检查下载水位，也不生成消费者的完成标记。自动发布请使用
`build_ready_fengshun_days`；不要先用单日入口写入自动消费者的最终输出路径。

## 自动构建 cron

保留已有 Fengshun 下载任务，另加每30分钟检查一次的构建任务。
先同步本仓库改动、确认附件并通过 dry-run，然后准备日志目录：

```bash
mkdir -p /home/xmetai/logs/xmetai-era5-fengshun
crontab -e
```

新增完整一行（时间按服务器本地时区）：

```cron
*/30 * * * * cd /home/xmetai/xmetai-tongyun-fengshun-dataset-building && /usr/bin/flock -n /tmp/xmetai-era5-fengshun-zarr.lock /home/xmetai/anaconda3/envs/xmetai/bin/python -u -m examples.dataset_building.build_ready_fengshun_days --config examples/dataset_building/fengshun_daily_server.yaml --write >> /home/xmetai/logs/xmetai-era5-fengshun/zarr-build.log 2>&1
```

锁与 Fengshun 下载锁及 Tongyun 构建锁分开；同一构建任务同时只运行一个实例。
自动消费者读取 `raw/_records/fengshun_c76_daily/watermark.json` 的 `latest_complete_day`，
只检查水位及之前的本地温度文件候选日，并核对当日16份文件。
水位是上限，不证明此前日期连续完整；补下的较早日期也会被发现并按序构建。
下载失败或缺日仍由下载器补数，构建器不会代替下载。
新版下载器已按水位逐日续跑：某日失败时停止推进，下次先补同一日，成功后继续追赶；
`--lag-days 6` 是允许请求的最晚日期，默认每次最多追赶10天。
本构建器的ZIP读取、日统计加工、完成标记与cron调用方式不因此改变。

构建先写入 `zarr/_attempts/<随机ID>/<产品名>`，只有实际写入及 Quick 校验通过后才移动到
最终目录，并原子写入 `zarr/_completed/YYYYMMDD.json`。完成标记和最终目录同时存在才跳过。
失败不记完成，保留 attempt 和报告，下次新建 attempt 重试；注意失败目录的磁盘占用。
若最终目录已存在但缺标记，拒绝覆盖，需人工检查报告和产物后恢复。
Quick 校验不等于完整发布验收；本流程尚不自动调用推理。

```bash
tail -f /home/xmetai/logs/xmetai-era5-fengshun/zarr-build.log
cat /home/xmetai/era5_download/era5_fengshun_c76/raw/_records/fengshun_c76_daily/watermark.json
ls /home/xmetai/era5_download/era5_fengshun_c76/zarr/_completed/
```

`WAIT` 表示水位不存在或文件未齐，`BUILD` 表示开始，`READY` 表示已发布，
`SKIP` 表示已经完成，`FAILED` 时查报告与 `_attempts`。
