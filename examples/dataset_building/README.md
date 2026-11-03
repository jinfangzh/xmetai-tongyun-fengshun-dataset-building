# 标准数据集构建

本目录提供可直接运行和调试的构建、追加与校验示例，自定义加工写在 `example_build_*.py` 中，通用流程由 `pipelines.dataset` 执行：

```text
加载契约与源数据 → process → 标准元数据组装 → 产物检查与写出 → 落盘 quick 校验
```

## 运行构建用例

修改用例末尾 `__main__` 中的契约、数据和附件路径，然后在仓库根目录运行；也可直接在 IDE 中设置断点调试。

```bash
python -m examples.dataset_building.example_build_era5
python -m examples.dataset_building.example_build_oras5
```

默认正式构建且不覆盖已有输出。在脚本中设置 `dry_run = True`，执行加工与内存产物检查，不写出、不执行落盘校验。

| 用例 | 目标规格 | 专属加工 |
| --- | --- | --- |
| `example_build_era5.py` | `era5_c78_p25_h6` | 风速合成、通道选择、单位换算、降水非负裁剪与 log1p、归一化；辐射累计窗口为 6 小时 |
| `example_build_oras5.py` | `oras5_c88_p25_h24` | 通道选择与归一化 |

两个用例均支持外部 mean/std；未提供时，要求源时间与目标完整一致，在归一化前按通道沿 time/lat/lon 计算（忽略 NaN，ddof=0）。统计量须符合目标契约。当前用例的 `mask_path` 必须提供。

## 开发自定义加工

在本目录新增 `example_build_<源数据>.py`，定义：

```python
def process(dataset, source_contract, spec, *, mean_path=None, std_path=None, mask_path=None):
    # 加工数据、准备附件，并设置目标 dtype、坐标和通道顺序。
    ...
    return product  # xarray.Dataset，包含 data、mean、std、mask
```

`process` 负责单位换算、派生变量、重采样、归一化及附件准备。使用 `datakit.io` 读取附件、`datakit.transforms` 组合基础算子；自行打开的附件应在关闭前加载所需数据。

pipeline 负责源资源管理、标准元数据组装、检查、写出和回读校验，不修复加工结果或自动执行目标契约中的预处理声明。返回数组的类型、布局、坐标和通道顺序须满足契约；统计量应对应归一化前的数据域。

| 基础算子 | 职责 |
| --- | --- |
| `compute_mean_std` | 按指定维度计算 mean/std，使用 float64 归约 |
| `apply_scale` | 按显式变量系数缩放 |
| `apply_log1p` | 对指定变量执行 log1p，不自动裁剪 |
| `apply_zscore` | 使用对齐的 mean/std 执行归一化 |

源契约参考 [`config/data_sources`](../../config/data_sources)，目标契约参考 [`config/dataset_specs`](../../config/dataset_specs)。选择与源单位、累计窗口及目标语义匹配的加工脚本；目标契约和输出路径须匹配本次产物的覆盖范围、网格、通道及身份。

## Python 任务接口

示例直接调用 `pipelines.dataset` 中的任务接口。需要在自己的程序中组织构建时，通过 `partial` 绑定加工参数：

```python
from functools import partial
from pipelines.dataset import build_zarr
from examples.dataset_building.example_build_era5 import process

report = build_zarr(
    source_config="/path/to/source.yaml",
    target_spec="/path/to/target.yaml",
    source_path="/data/input.nc",
    destination="/data/output/product.zarr",
    process=partial(process, mean_path="/data/mean.nc", std_path="/data/std.nc", mask_path="/data/mask.nc"),
    dry_run=False,
)
print(report.to_dict())
```

具体用例在 `__main__` 中设置参数并调用实际加工函数；正常导入不执行任务。

## 追加与校验

修改 [example_append_era5.py](example_append_era5.py)、[example_validate_era5.py](example_validate_era5.py) 中的路径后，可直接运行或调试：

```bash
python -m examples.dataset_building.example_append_era5
python -m examples.dataset_building.example_validate_era5
```

也可直接调用：

```python
from pipelines.dataset import append_zarr, validate_zarr, compare_zarr

# 已有 B + 后续 A → 独立 C，内部执行完整校验。
report = append_zarr(
    source="/data/A.zarr", target="/data/B.zarr", destination="/data/C.zarr",
    spec="/path/to/combined_target.yaml", output="/data/reports/append.yaml",
)
print(report.release_passed)

report = validate_zarr(
    target="/data/C.zarr", spec="/path/to/combined_target.yaml",
    mode="full", output="/data/reports/validation.yaml",
)
print(report.release_passed)

comparison = compare_zarr(source="/data/reference.zarr", target="/data/C.zarr")
print(comparison["equal"])
```

追加按 B + A 生成独立 C，保留输入且拒绝已有输出，写出后内部执行 full 校验。契约须描述完整 BA，并准备相应外部附件及发布证据。追加期间输入保持静止，预留完整 BA 的额外磁盘空间；内存至少容纳一个时间片及静态附件。

独立校验只检查目标产物与契约及声明规则，不执行源到目标的加工对照；默认 full，按 `release_passed` 判断通过；quick 按 `quick_passed` 判断，不代替完整验收。`output=None` 不保存报告；指定校验报告时仅接受新的 `.yaml` 路径。

接口参数、模块职责、资源管理及失败处理统一见 [数据集任务接口](../../pipelines/dataset/README.md)。
