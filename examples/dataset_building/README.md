# 数据集制作示例

当前保留的具体制作流程位于 [history](history/README.md)：

| 示例 | 用途 |
| --- | --- |
| `history/era5_nc_to_zarr.py` | ERA5 日 NetCDF、单位预处理、归一化与 Zarr 写出 |
| `history/oras5_nc_to_zarr_88.py` | ORAS5 日数据组装、归一化与 Zarr 写出 |
| `history/validate_standard_conversion.py` | 按转换规则对照源数据与目标产物 |

在仓库根目录查看参数：

```bash
python -m examples.dataset_building.history.era5_nc_to_zarr --help
python -m examples.dataset_building.history.oras5_nc_to_zarr_88 --help
```

历史制作脚本会删除已有目标目录后重新生成数据，运行前应确认 `--destination`。
具体输入要求和制作命令见 [history/README.md](history/README.md)。

创建、追加、验收、比较直接使用 `pipelines.dataset` 的 Python API，
参数与调用方式见 [流程说明](../../pipelines/dataset/README.md)。
