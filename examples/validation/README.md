# NetCDF/Zarr 基础数值对照

本目录比较已有文件的坐标、通道和数值，辅助检查转换或导出的结果。
按固定 YAML 标准执行 quick/full 验收，请使用
`pipelines.dataset.validate_zarr`；output=None 时只返回报告。
按显式转换规则生成保真报告，参见 [dataset_building](../dataset_building/README.md)。

| 入口 | 比较对象 | 输出 |
| --- | --- | --- |
| `validate_netcdf.py` | 转换后的 NetCDF 与参考 NetCDF | 基础质量诊断、共同坐标范围的 bias/MAE/RMSE，可选逐通道 CSV |
| `validate_zarr_nc.py` | 归一化源 Zarr 与导出的 NetCDF | 反归一化后的结构、坐标、通道及数值对照，输出 JSON |

在仓库根目录运行：

```bash
python -m examples.validation.validate_netcdf \
  --converted /data/output/converted.nc --reference /data/reference.nc \
  --variable data --channel-report-csv outputs/comparison/channels.csv

python -m examples.validation.validate_zarr_nc \
  --zarr-path /data/era5.zarr --nc-path /data/output/converted.nc \
  --report-json outputs/comparison/zarr_nc.json
```

`validate_netcdf.py` 支持旧式 HDF5 存储的 NetCDF 回退读取；该格式需要 `h5py`，可安装
`.[io]`。它在共同坐标范围比较，不据此证明完整覆盖。`validate_zarr_nc.py` 使用源数据中的
统计量反归一化。两者都不提供标准产物的发布验收结论。

公共 Python 函数从 `datakit.validation` 或 `pipelines.dataset` 导入；本包只提供任务脚本。
标准产物的校验接口见 [数据集工作流](../../pipelines/dataset/README.md)。
