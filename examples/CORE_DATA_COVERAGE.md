# xmetai-core 数据场景示例覆盖

这份清单把 `xmetai-core` 中仍能识别出的通用数据场景映射到 datakit 示例。映射按
能力而不是旧文件逐行迁移：示例只调用 datakit 公共 API，不导入 core，也不复制
`argparse`、日志、守护进程或训练框架代码。

## `xmetai/utils/reanalyse.py`

| core 能力 | datakit 公共入口 | 示例 |
|---|---|---|
| 维度、坐标和变量名整理 | `datakit.transforms` | `process_xarray.py` |
| 层与通道转换、归一化和单位缩放 | `datakit.transforms` | `process_xarray.py` |
| 累计量转增量、重采样和预报步聚合 | `datakit.transforms` | `process_xarray.py` |
| 空间裁剪和插值 | `datakit.transforms` | `process_xarray.py` |
| mean/std、差分、时次气候态和分位数 | `pipelines.dataset` | `statistics_summary.py` |
| NaN/Inf、范围、完整年份和 schema 检查 | `datakit.validation` | `statistics_summary.py` |
| NetCDF、GRIB、NumPy 和 Zarr 本地读写 | `datakit.io` | `io_local.py`、`convert_to_zarr.py` |
| 按年保存、统计产物和 Zarr 导出 | `pipelines.dataset` | `zarr_products.py` |

`get_trend`、`detrend_single`、概率预报后处理、模型掩码策略和训练专用通道权重不属于
当前 datakit 的基础能力闭环，因此不在示例中重建。

## `tools/`

| core 工具 | datakit 等价能力 | 示例 |
|---|---|---|
| `calculate_zarr_mean.py` | `build_slot_climatology` | `zarr_products.py` |
| `check_zarr_range.py` | `check_zarr_range` | `zarr_products.py` |
| `compute_quantile.py` | `compute_quantiles`、`build_month_day_quantiles` | `statistics_summary.py`、`zarr_products.py` |
| `extract_nc_from_zarr.py` | `extract_zarr_to_netcdf` | `zarr_products.py` |
| `ensemble_nc_to_zarr.py` | `convert_ensemble_netcdf_to_zarr` | `ensemble_to_zarr.py` |
| `radar_bins_to_zarr.py` | `convert_cinrad_to_zarr` | `radar_to_zarr.py` |
| `export_radar_image.py` | `quality_control_reflectivity`、雷达可视化 | `radar_quality.py`、`visualize.py` |
| `visualize_pred_gif.py` | `plot_field`、`save_field_gif` | `visualize.py` |

GPU 分位数、目录监控、进程池调度、删除源文件、交互式参数选择和 CLI 输出属于上层
执行策略，不属于 datakit 的数据算法，也不纳入示例。

## 验证层次

- `pytest tests/test_examples.py` 会执行无外部数据、无可选 I/O 引擎的示例，并检查
  所有示例都不依赖 `xmetai`、Torch 或远程存储库。
- Zarr、NetCDF、集合和雷达数据集示例使用临时合成数据，可在安装声明依赖后端到端
  运行；不会下载数据，也不会在仓库中留下产物。
- `tests/test_pipelines.py` 对路径级 pipeline 的调度、写入参数和关闭语义进行隔离测试。

