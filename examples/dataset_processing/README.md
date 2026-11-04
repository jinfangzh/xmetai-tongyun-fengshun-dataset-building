# NetCDF 预处理、提取与导出

本目录处理已有资料：整理原始字段、转换单位、提取指定日期，或导出 Zarr 切片。
各脚本的输入布局不同，请按实际处理阶段选择：

| 入口 | 输入 | 输出与处理 |
| --- | --- | --- |
| `process_era5_pl_nc.py` | 按变量组织的 ERA5 日高空 NetCDF | 选择 13 个气压层、调整网格，输出 channel 形式的 NetCDF |
| `process_oras5.py` | `<year>/<YYYYMMDD>_oras5.nc` | 整理海洋变量和深度通道到指定网格，输出日 NetCDF；不做归一化 |
| `era5_convert_units.py` | channel 形式的日 NetCDF | 按显式累计时长换算单位，可选择通道；不做归一化 |
| `era5_extract_raw_truth_day.py` | 按组、变量和年份组织的 ERA5 月度原始归档 | 提取一个 UTC 日，输出指定通道的实况 NetCDF |
| `era5_zarr_to_nc.py` | 已归一化 Zarr 及其 mean/std | 提取时间和通道、反归一化后导出 NetCDF，可再按通道和日期拆分 |

制作归一化 Zarr 和按产品 YAML 验收见
[dataset_building](../dataset_building/README.md)。同为 NetCDF 的文件可能具有不同变量、
通道和单位约定，不能仅凭文件格式串联这些脚本；尤其不要对已换算单位的资料重复执行预处理。

## 运行方式

在仓库根目录安装所需的 `netcdf`、`parallel`、`interpolation` extras，使用模块入口。
例如，对一个 UTC 日提取少量实况通道：

```bash
python -m examples.dataset_processing.era5_extract_raw_truth_day \
  --source /data/era5/raw --date 2025-01-01 \
  --output /data/output/truth_20250101.nc --subset demo-6
```

对已有归一化数据集导出切片：

```bash
python -m examples.dataset_processing.era5_zarr_to_nc \
  --zarr-path /data/era5.zarr --save-path /data/output/era5_20250101.nc \
  --start-date 2025-01-01 --time-interval 1 --channels t2m,z500
```

`process_oras5.py` 带有服务器路径和历史日期默认值，运行时显式指定输入、输出和日期：

```bash
python -m examples.dataset_processing.process_oras5 \
  --input-root /data/oras5/raw --output-root /data/oras5/channels \
  --start-date 2025-01-01 --end-date 2025-01-02 --grid-res 0.25
```

其余参数及各脚本的实际输入布局见模块说明与 `--help`。
