# 具体数据工作流与运行入口

本目录将仓库能力应用到具体数据任务：从源数据下载、接入和处理，到产物制作、质量校验，
再到不同来源预报数据的评测与报告。任务脚本调用 `pipelines` 组织流程，
并复用 `datakit` 的基础组件。源数据声明独立位于仓库根目录 `config/data_sources/`。

使用现有任务时先选择下面的目录，查看对应 README 和配置，再准备实际数据。

先在仓库根目录安装项目：

```bash
python -m pip install -e .
```

需要对应可选能力时安装 extras：

```bash
python -m pip install -e '.[netcdf]'         # NetCDF engines
python -m pip install -e '.[io]'             # All I/O engines
python -m pip install -e '.[interpolation]'     # SciPy interpolation
python -m pip install -e '.[visualization]'  # Matplotlib and Pillow
python -m pip install -e '.[all,test]'       # 数据工作流及测试依赖
```

## 目录

| 目录 | 内容 |
|---|---|
| [dataset_io](dataset_io/README.md) | `io_example.py` 展示统一调用；根目录 `config/data_sources/` 保存独立 YAML 声明 |
| [dataset_building](dataset_building/README.md) | history 下保留 ERA5/ORAS5 制作和转换对照流程 |
| [dataset_processing](dataset_processing/README.md) | NetCDF 预处理、单位换算、原始实况提取和 Zarr 切片导出 |
| [validation](validation/README.md) | 比较两份 NetCDF，或对照 Zarr 与导出 NetCDF 的数值 |
| [evaluation](evaluation/README.md) | 对预报评分并生成报告 |
| `dataset_downloader/` | CDS 和 Copernicus Marine 下载方案 |

可先查看脚本参数：

```bash
python -m examples.dataset_building.history.era5_nc_to_zarr --help
python -m examples.dataset_processing.era5_zarr_to_nc --help
```

任务详情见 [数据集接入](dataset_io/README.md)、[制作](dataset_building/README.md)、
[处理与导出](dataset_processing/README.md)、
[标准验收](../pipelines/dataset/README.md)、[基础对照](validation/README.md) 和
[评测与报告](evaluation/README.md)。

`dataset_building` 与 `validation` 的包入口不转导出公共函数；Python 调用直接引用
`pipelines.dataset`、`pipelines.dataset` 或 `datakit.validation`。

制作和处理脚本需要实际数据。下载器按照 Batch YAML 中的 `storage.root_path`
写入文件，运行前应检查数据范围、输出目录和服务凭据。

## 预报数据评测任务

在仓库根目录安装评测依赖后，可按配置名运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
```

也可把 `--config` 指向 `examples/evaluation/configs/` 下的具体 Python 文件。

## 数据下载方案

| 目录 | 内容 | 安装与运行说明 |
|---|---|---|
| `dataset_downloader/cds_download/` | CDS / ERA5 下载与 Batch 配置 | [CDS 下载说明](dataset_downloader/cds_download/README.md) |
| `dataset_downloader/copernicusmarine_download/` | Copernicus Marine / ORAS5 下载与 Batch 配置 | [Copernicus Marine 下载说明](dataset_downloader/copernicusmarine_download/README.md) |

两套下载器分别使用所在目录的 `requirements.txt`；`datakit` 的安装 extras
不包含 `cdsapi` 或 `copernicusmarine`。
