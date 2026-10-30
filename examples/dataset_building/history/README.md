# 历史数据集制作与转换验收入口

本目录保留 ERA5、ORAS5 各自的完整制作流程，以及转换专项验收入口。

| 目标 | 入口 | 输入与输出 |
| --- | --- | --- |
| 制作 ERA5 归一化数据集 | `era5_nc_to_zarr.py` | 日 NetCDF、mean/std、mask → 归一化 Zarr |
| 制作 ORAS5 88 通道数据集 | `oras5_nc_to_zarr_88.py` | 通道 NetCDF、mean/std、mask → 归一化 Zarr |
| 检查转换数值保真 | `validate_standard_conversion.py` | 源、目标和转换规则 JSON → 转换对照报告 |

## 从 NetCDF 制作归一化 Zarr

在仓库根目录运行，按实际资料指定路径。两个制作脚本都需要与所选通道匹配的统计量和 mask：

```bash
python -m examples.dataset_building.history.era5_nc_to_zarr \
  --source /data/era5/daily --destination /data/output/era5.zarr \
  --mean-nc /data/era5/mean.nc --std-nc /data/era5/std.nc \
  --mask-nc /data/era5/mask.nc --accumulation-hours 6

python -m examples.dataset_building.history.oras5_nc_to_zarr_88 \
  --source /data/oras5/channels --destination /data/output/oras5.zarr \
  --mean-nc /data/oras5/mean.nc --std-nc /data/oras5/std.nc \
  --mask-nc /data/oras5/mask.nc
```

这些制作入口按脚本参数写出产物，不读取目标 YAML 来保证符合完整标准。
不要对已归一化的 Zarr 重复执行归一化。脚本中的默认路径需要替换，
并在运行前确认目标目录：已有目标目录会被删除后重新生成。
原始 NetCDF 预处理、单位转换和导出切片见
[dataset_processing](../../dataset_processing/README.md)。

## Standard dataset validation

具体产品的标准 YAML 位于 [`config/dataset_specs`](../../../config/dataset_specs)，不随基础库安装包分发；
通用模型与严格加载器位于 `datakit.schema`，由调用方显式传入 YAML 路径。
数据集验收直接调用公共 Python 函数：

```python
from pipelines.dataset import validate_zarr

report = validate_zarr("/data/land.zarr", "config/dataset_specs/era5_land_c8_1p5_h24.yaml")
# 需要保存时显式指定文件。
report = validate_zarr(
    "/data/land.zarr", "config/dataset_specs/era5_land_c8_1p5_h24.yaml",
    output="/path/outside/data/report.json",
)
```

`output=None` 不保存文件；指定路径时保存单份 JSON，已有文件拒绝覆盖。
默认完整检查；抽样时传 `mode="quick"`。有冻结证据时在外围使用 load_validation_run 加载 spec。
转换专项的 `validate_standard_conversion.py` 保留独立入口。

Conversion JSON configurations reference YAML through `target_schema`. Relative paths
are resolved against the containing configuration, including evidence paths.
Reports and independent evidence remain JSON. A quick pass verifies the
selected main-data blocks and complete metadata/coordinate checks. A full scan
requires `--mode full` without `--max-blocks`, plus the applicable physical
coverage and production completion evidence. See the
[validation guide](../../../docs/DATASET_VALIDATION.md) for report fields and tests.

For a frozen run, use `--run-config path/to/run.json` instead of `--schema`.
This JSON references the unchanged YAML and supplies an `execution` object with
`snapshot_id`, receipt `{path, sha256}` references and applicable requirements.
Runtime production/conversion receipts are not accepted inside YAML.
Adding a receipt to execution JSON does not change the YAML hash bound by it.
Conversion configurations also accept this
nested `execution` object. A run cannot disable a schema-required evidence gate.

Core reader probes and consumer/model receipt checks have been removed. The
validation functions cover data format, integrity, quality and conversion fidelity,
not model compatibility. Retired
`consumer_required`, `consumer_evidence` and `model_evidence` settings are rejected.
JSON reports no longer expose a consumer-compatibility conclusion.

Combination validation, its CLI and configuration template have also been removed.
Call validate_zarr once per dataset from an outer loop; no internal batch runner
or joint-use conclusion is provided. Tool version `1.8` removes the
`cross_dataset_compatibility` report category and `combination.scope` check.
Historical reports are not rewritten.

Program regression tests live in `tests/dataset_building/acceptance/`; they use small fixtures
and injected faults. Passing these tests does not establish that server data
has passed validation.
