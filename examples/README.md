# Examples

本目录包含使用小型合成数据的 `datakit` 示例，以及需要外部服务凭据的独立
数据下载方案。下载器不是 `datakit` 的公共 API。与 `xmetai-core/utils`、`xmetai-core/tools` 的能力对应关系见
[CORE_DATA_COVERAGE.md](CORE_DATA_COVERAGE.md)。

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

## 示例清单

| 文件 | 内容 |
|---|---|
| `io_local.py` | NPY 本地读写 |
| `process_xarray.py` | 坐标规范化、空间选择和描述统计 |
| `convert_to_zarr.py` | NumPy 数据通过 dataset pipeline 转为 Zarr |
| `statistics_summary.py` | mean/std 和时次气候态 |
| `radar_quality.py` | 合成雷达反射率质量控制 |
| `visualize.py` | 将二维气象场保存为 PNG |
| `zarr_products.py` | Zarr 检查、统计产物、气候态、分位数及 NetCDF 导出 |
| `ensemble_to_zarr.py` | 合成集合 NetCDF 成员组装为 Zarr |
| `radar_to_zarr.py` | 合成 CINRAD 二进制序列制作 Zarr 数据集 |
| `evaluation/configs/` | Fengqing 评测的具体数据路径、时间和输出配置 |

运行方式：

```bash
python examples/io_local.py
python examples/process_xarray.py
python examples/statistics_summary.py
python examples/radar_quality.py
```

路径级数据集示例需要兼容的本地 I/O 后端：

```bash
python examples/convert_to_zarr.py
python examples/zarr_products.py
python examples/ensemble_to_zarr.py
python examples/radar_to_zarr.py
python examples/visualize.py
```

上述合成数据示例使用临时目录。下载器会按照 Batch YAML 中的 `storage.root_path`
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
