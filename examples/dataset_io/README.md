# YAML 数据集访问示例

`io_example.py` 用一个 `main()` 展示加载 YAML、定位源文件、读取、可选标准化与导出。
具体数据集声明在仓库根目录的 [config/data_sources](../../config/data_sources/)，
本目录不再维护数据源 Python 包装模块，评测流程也不导入示例代码。

## 运行

在仓库根目录安装所需依赖，例如 `.[io,parallel]`，并替换真实数据路径：

```bash
python -m examples.dataset_io.io_example --help

python -m examples.dataset_io.io_example \
  --config config/data_sources/forecast_fuxi_ensemble.yaml \
  --path /data/fuxi --init-time 2025-01-01 --lead-time 6 12 --member 0 \
  --variables z500 t2m --output /data/output/fuxi.nc

python -m examples.dataset_io.io_example \
  --config config/data_sources/observation_diamond_flat.yaml \
  --path /data/stations --time 2025-01-01T06:00 --variables tp

python -m examples.dataset_io.io_example \
  --config config/data_sources/climatology_cra.yaml \
  --path /data/cra_climatology --cycle-time 010106
```

省略 `--variables` 时保留源变量、单位和缺测值。提供变量时才显式调用标准化；
配置没有 mapping 时仅按原始变量名选择。按目录组织的数据需要选择条件或显式传入
`--all-files`。日序气候态的 `--path` 指向单个 NetCDF。时间条件遵循源时区，
不隐式转成 UTC。控制台显示结构；`--output` 在关闭源之前写出结果，拒绝覆盖已有输出。

## Python 调用

```python
from datakit import io
from datakit.schema.mapping import normalize_dataset

config = "config/data_sources/forecast_fuxi_ensemble.yaml"
description = io.get_description(config)
source = io.open_dataset(
    "/data/fuxi", description,
    selection={"init_time": ["2025-01-01"], "lead_time": [6, 12]},
)
with source.read() as raw:
    standard = normalize_dataset(raw, ["z500", "t2m"], description).load()
```

也可以用 `io.source_config("forecast_fuxi_ensemble")` 定位随项目分发的 YAML；
支持通过 `directory=` 显式选择配置目录。自定义 YAML 直接传路径即可，不需要注册 Python 模块。

通用声明与后端边界见 [配置说明](../../config/README.md)。
评测的单位口径、GRIB 指标变量筛选、日历对齐、平滑和插值仍由
`pipelines/evaluation/sources/` 显式组织。产品验收 YAML 位于
`examples/dataset_building/configs/standards/`，与这里的源数据访问声明用途不同。
