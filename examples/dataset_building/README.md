# 数据集制作与标准验收

本目录放置生成数据集和验收构建结果的任务入口。按下面的输入与目标选择脚本：

| 目标 | 入口 | 输入与输出 |
| --- | --- | --- |
| 制作 ERA5 归一化数据集 | `era5_nc_to_zarr.py` | 日 NetCDF、mean/std、mask → 归一化 Zarr |
| 制作 ORAS5 88 通道数据集 | `oras5_nc_to_zarr_88.py` | 通道 NetCDF、mean/std、mask → 归一化 Zarr |
| 按产品标准验收 | `validate_standard_dataset.py` | 数据集、YAML 标准或执行 JSON → quick/full 报告 |
| 检查转换数值保真 | `validate_standard_conversion.py` | 源、目标和转换规则 JSON → 转换对照报告 |

预处理原始 NetCDF、单位转换和导出切片见
[dataset_processing](../dataset_processing/README.md)；基础 NetCDF/Zarr 数值对照见
[validation](../validation/README.md)。本包不转导出公共函数，程序调用使用
`pipelines.dataset` 的转换/提取接口及 `pipelines.dataset_building` 的验收接口。

## 从 NetCDF 制作归一化 Zarr

在仓库根目录运行，按实际资料指定路径。两个制作脚本都需要与所选通道匹配的统计量和 mask：

```bash
python -m examples.dataset_building.era5_nc_to_zarr \
  --source /data/era5/daily --destination /data/output/era5.zarr \
  --mean-nc /data/era5/mean.nc --std-nc /data/era5/std.nc \
  --mask-nc /data/era5/mask.nc --accumulation-hours 6

python -m examples.dataset_building.oras5_nc_to_zarr_88 \
  --source /data/oras5/channels --destination /data/output/oras5.zarr \
  --mean-nc /data/oras5/mean.nc --std-nc /data/oras5/std.nc \
  --mask-nc /data/oras5/mask.nc
```

这些制作入口按脚本参数写出产物，不读取目标 YAML 来保证符合完整标准；
需要标准产物时，另行执行下方验收。不要对已归一化的 Zarr 重复执行归一化。
更多选择参数先运行相应模块的 `--help`；脚本中的服务器路径默认值需要替换。

## Standard dataset validation

具体产品的标准 YAML 位于 [`configs/standards`](configs/standards)，不随基础库安装包分发；
通用模型与严格加载器位于 `datakit.schema.datasets`，由调用方显式传入 YAML 路径。
转换对照配置模板位于本目录的 `configs/`。两个验收入口负责参数、信号和退出码，
检查、扫描、证据绑定及报告由
[`pipelines.dataset_building`](../../pipelines/dataset_building/README.md) 实现：

- `validate_standard_dataset.py`：按一份或多份独立 YAML 检查数据集。
- `validate_standard_conversion.py`：按显式规则比较冻结的源与目标。

Run from the repository root:

```bash
python -m examples.dataset_building.validate_standard_dataset \
  --schema-directory examples/dataset_building/configs/standards \
  --mode quick --sample-blocks 3 --output /path/outside/data/reports
```

For one dataset, use `--schema examples/dataset_building/configs/standards/era5_land_c8_1p5_h24.yaml`.
Multiple paths may follow `--schema`; do not combine multi-dataset selection with
a single `--dataset-path` override. Old profile/inventory options are removed.
Conversion JSON configurations reference YAML through `target_schema`. Relative paths
are resolved against the containing configuration, including evidence paths.
Reports and independent evidence remain JSON. A quick pass verifies the
selected main-data blocks and complete metadata/coordinate checks. A full scan
requires `--mode full` without `--max-blocks`, plus the applicable physical
coverage and production completion evidence. See the
[validation guide](../../docs/DATASET_VALIDATION.md) for report fields and tests.

For a frozen run, use `--run-config path/to/run.json` instead of `--schema`.
This JSON references the unchanged YAML and supplies an `execution` object with
`snapshot_id`, receipt `{path, sha256}` references and applicable requirements.
Runtime production/conversion receipts are not accepted inside YAML.
Adding a receipt to execution JSON does not change the YAML hash bound by it.
Conversion configurations also accept this
nested `execution` object. A run cannot disable a schema-required evidence gate.

Core reader probes and consumer/model receipt checks have been removed. The
two validation CLIs cover data format, integrity, quality and conversion fidelity,
not model compatibility. Retired
`consumer_required`, `consumer_evidence` and `model_evidence` settings are rejected.
JSON/CSV/Markdown reports no longer expose a consumer-compatibility conclusion.

Combination validation, its CLI and configuration template have also been removed.
Selecting multiple YAMLs or run configurations still validates each dataset
independently; it does not impose shared time, grid or mask requirements or
produce a joint-use conclusion. Tool version `1.8` removes the
`cross_dataset_compatibility` report category and `combination.scope` check.
Historical reports are not rewritten.

Program regression tests live in `tests/dataset_building/acceptance/`; they use small fixtures
and injected faults. Passing these tests does not establish that server data
has passed validation.
