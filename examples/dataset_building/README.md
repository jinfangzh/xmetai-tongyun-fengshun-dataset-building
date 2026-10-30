# 数据集制作示例

`standard_pipeline.py` 提供“两份契约、两个路径”的构建示例，适用于显式声明的
ERA5/ORAS5 packed NetCDF 输入。它读取源配置的 IO 布局，通过显式选择的 ERA5 或 ORAS5
处理函数转换数据、匹配统计量、组装元数据，再调用 `create_zarr` 检查和写出；不覆盖已有输出。

```bash
python -m examples.dataset_building.standard_pipeline \
  --processor era5 \
  --source-config /path/to/source.yaml \
  --target-spec /path/to/target.yaml \
  --source /data/input.nc \
  --destination /data/output/result.zarr \
  --mean-nc /data/stats/mean.nc \
  --std-nc /data/stats/std.nc \
  --mask-nc /data/stats/mask.nc \
  --dry-run
```

源 YAML 使用 `config/data_sources` 的格式；单文件 packed NetCDF 可以声明
`name: packed_nc`、`format: netcdf`、`layout: {}`、`example: input.nc`。
目标 YAML 使用 `config/dataset_specs` 的格式，须与本次时间覆盖、通道、网格和身份一致。
数据路径由命令行指定，目标文件名仍须满足目标规范的目录身份要求。
源值须满足目标声明的源单位和累计窗口；此示例不自动推断这些语义，也不自动执行源 mapping，
避免与目标预处理重复换算。

Python 入口只接收两份契约、两个路径、处理函数及可选的 `dry_run`：

```python
from functools import partial
from examples.dataset_building.standard_pipeline import build_dataset, process_era5

process = partial(
    process_era5,
    mean_path="/data/stats/mean.nc",
    std_path="/data/stats/std.nc",
    mask_path="/data/stats/mask.nc",
    accumulation_hours=6,
)
report = build_dataset(
    "source.yaml", "target.yaml", "/data/input.nc", "/data/output/result.zarr",
    process=process,
    dry_run=True,
)
```

阶段入口：`load_source_contract` → `load_target_contract` → `validate_source` → `read_source` → `process(raw, spec)` → `write_product` → `validate_product`。
`process` 返回完整的目标 Dataset，统计量和掩膜由处理函数加载、匹配并组装，pipeline 不管理这些参数。
内置 `process_era5` 明确执行风速合成、单位预处理和归一化；`process_oras5` 只选择通道并归一化。
两者检查目标处理声明与自身算法一致，不解释任意 YAML 运算。ORAS5 使用 `--processor oras5`。
其他时间/空间/变量操作在自定义处理函数中明确编写；特殊文件组织由源 IO 配置描述。

写出前 `create_zarr` 执行内存产物检查，成功写出后最后调用 `validate_zarr(mode="quick")` 检查落盘数据和声明的外部附件。
普通运行返回 quick 校验报告，CLI 以 `quick_passed` 判断成功；写出失败返回创建报告。
`dry_run` 不写出，返回内存产物的创建检查报告。quick 检查不等于完整扫描或发布放行。
目标规范要求的 const/weight 等外部附件仍需由任务准备，本例只组装 mean/std/mask。
`validate_source(source_path, source_config)` 在处理前逐文件检查路径、布局匹配、可读性、
坐标标签唯一性、显式 packed 结构，以及已出现映射变量的源单位。错误抛异常，停止后续处理。
mapping 是可选映射，不作为必需变量清单；单位只进行声明字符串的严格比较，不自动换算。
该接口不扫描主数据值，也不证明未声明的日期覆盖、时间语义或物理范围；返回摘要明确检查范围。

默认读取惰性访问数据，但不提供流式追加或强制内存预算，先用于小样本。
mean/std 应对应 zscore 前的数据域；log1p 通道的有效统计量必须符合目标声明。
旧 ORAS5 的 `so19979` 等通道不能自动视为现行规范的 `os19978`。

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
