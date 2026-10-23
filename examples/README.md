# Examples

本目录提供具体数据集的接入、制作、处理、校验和评测脚本，以及独立的数据下载方案。
下载器不是 `datakit` 的公共 API。

先在仓库根目录安装项目：

```bash
pip install -e .
```

需要对应可选能力时安装 extras：

```bash
pip install -e '.[netcdf]'         # NetCDF engines
pip install -e '.[io]'             # All I/O engines
pip install -e '.[interpolation]'     # SciPy interpolation
pip install -e '.[visualization]'  # Matplotlib and Pillow
pip install -e '.[all]'            # 运行全部文件格式和可视化示例
```

## 目录

| 目录 | 内容 |
|---|---|
| `dataset_io/` | 数据组织说明、变量映射和原始访问 |
| `dataset_building/` | ERA5、ORAS5 等数据集制作脚本 |
| `dataset_processing/` | Zarr 导出、反归一化及 NetCDF 拆分 |
| `validation/` | 数据校验配置和运行脚本 |
| `evaluation/` | 评测配置和运行脚本 |
| `dataset_downloader/` | CDS 和 Copernicus Marine 下载方案 |

可先查看脚本参数：

```bash
python examples/dataset_building/era5_nc_to_zarr.py --help
python examples/dataset_processing/era5_zarr_to_nc.py --help
```

制作和处理脚本需要实际数据。下载器按照 Batch YAML 中的 `storage.root_path`
写入文件，运行前应检查数据范围、输出目录和服务凭据。

## Fengqing 评测应用

在仓库根目录安装评测依赖后，可按配置名运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
```

也可把 `--config` 指向 `examples/evaluation/configs/` 下的具体 Python 文件。

## 数据下载方案

| 目录 | 内容 | 安装与运行说明 |
|---|---|---|
| `dataset_downloader/cds_download/` | CDS / ERA5 下载与 Batch 配置 | [CDS 下载说明](dataset_downloader/cds_download/README_download.md) |
| `dataset_downloader/copernicusmarine_download/` | Copernicus Marine / ORAS5 下载与 Batch 配置 | [Copernicus Marine 下载说明](dataset_downloader/copernicusmarine_download/README.md) |

两套下载器分别使用所在目录的 `requirements.txt`；`datakit` 的安装 extras
不包含 `cdsapi` 或 `copernicusmarine`。
