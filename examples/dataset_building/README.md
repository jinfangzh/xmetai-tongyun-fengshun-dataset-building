# 数据集制作工作流

本目录提供面向具体数据产品的制作任务。输入是源文件、已有数据集、统计量和掩膜等，
脚本按约定组织通道、单位、归一化和元数据，输出可供训练、后续处理或校验使用的数据集。
底层读写复用开源接口，公共步骤复用 `datakit` 和 `pipelines.dataset`。

## 选择任务

| 入口 | 输入与输出 |
| --- | --- |
| `era5_nc_to_zarr.py` | ERA5 NetCDF、均值、标准差和 mask → 预处理并归一化的 Zarr |
| `oras5_nc_to_zarr_88.py` | ORAS5 源数据及统计量 → 按所选通道制作的归一化 Zarr |
| `legacy_zarr_refresh/` | 已归一化的旧 Zarr → 通道、坐标及附属量对齐后的新数据集 |

旧 Zarr 整理是独立任务，处理约定仍由其配置描述，当前未纳入 `datakit` 公共 API。
说明和命令见 [旧 Zarr 整理](legacy_zarr_refresh/README.md)。

## 运行 ERA5 制作任务

在仓库根目录安装依赖，并将路径替换为实际输入：

```bash
python -m pip install -e '.[io,parallel]'
python -m examples.dataset_building.era5_nc_to_zarr \
  --source /data/era5/nc \
  --destination /data/output/era5.zarr \
  --mean-nc /data/era5/stats/mean.nc \
  --std-nc /data/era5/stats/std.nc \
  --mask-nc /data/era5/stats/mask.nc \
  --accumulation-hours 6
```

均值、标准差、通道和累计时长应与输入的物理含义对应。
该脚本会删除已有目标目录后重新生成产物，运行前应确认 `--destination`。
ORAS5 任务的参数可用以下命令查看：

```bash
python -m examples.dataset_building.oras5_nc_to_zarr_88 --help
```

## 检查产物并交给后续步骤

写出后，按实际产品选择 profile，运行 [数据校验任务](../validation/README.md)。
需要反归一化、时间提取或 NetCDF 导出时，使用 `examples/dataset_processing/` 下的任务。
代码修改可运行 `tests/test_era5_nc_to_zarr.py` 和 `tests/test_oras5_nc_to_zarr_88.py`。
