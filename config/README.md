# 数据源描述配置

`data_sources/` 只保存 YAML。每份声明独立描述一个数据集及其文件组织方式，
不导入 Python、不绑定机器路径，也不组织评测或验收。读取选择、根目录与输出位置由调用方提供。

| YAML | 文件组织 |
| --- | --- |
| `analysis_cra.yaml` | CRA 分日分析，高空和陆面分文件 |
| `climatology_cra.yaml` | CRA 月日时次 GRIB 气候态，无观测年份 |
| `climatology_station_probability.yaml` | 月日时次站点概率表，显式阈值列 |
| `forecast_fengqing.yaml` | 风清高空/地面分文件，成员在文件内部 |
| `forecast_fuxi.yaml` | 伏羲单模型，每个文件对应一个时效编号 |
| `forecast_fuxi_ensemble.yaml` | 伏羲集合，成员在目录中 |
| `observation_babj.yaml` | BABJ 台风报文，保留全部时效与额外列 |
| `observation_diamond.yaml` | Diamond 按日目录的站点文本 |
| `observation_diamond_flat.yaml` | Diamond 平铺站点文本 |

## 声明范围

- `name`、`example`、`notes`：标识、文件布局示例与数据含义说明。
- `layout`：目录模板、文件名、文件分组或匹配模式；空对象表示直接访问单文件/store。
- `format`：只有格式名称时直接写 `format: grib`；需要参数时写成包含 `name` 的映射。
  格式名称可为 `netcdf`、`grib`、`zarr`、`numpy`、
  `diamond_station`、`station_probability` 或 `babj`。
- 文本格式映射的其他字段：编码、表头和列定义。
- `notes`：位于文件末尾的 `|-` 多行说明，保留目录示例和换行。
- `lead_time`、`cycle`、`time_zone`：时效编号换算、周期和源时区。
- `structure`、`member`：记录维度及成员组织。
- `thresholds`：站点气候概率表各列对应的阈值。
- `mapping`：变量名称、单位及线性换算、坐标别名、打包字段和维度规则。

`format: grib` 与 `format: {name: grib}` 等价，IO 层校验后统一解释；
格式描述决定库内后端选择；不接受 Python 导入路径或
可执行表达式，未知格式会报错。旧 `reader` 字段不再接受。声明的容器格式参与实际读取与写出；
必要时调用方可显式通过 `read(format=...)` 覆盖，例如使用 NetCDF 小样本验证 GRIB 目录布局。
缺测值和源时间默认原样保留。标准化由调用方显式执行，不在 read() 中隐式发生。
新的文件格式需要先实现 IO 后端；已有格式的新布局可直接增加 YAML。

单文件日序气候态直接通过 `io.open(path).read()` 读取，不单独维护数据源 YAML。
评测适配器保留 data/level 打包结构的默认映射，并允许调用方覆盖。
读取保留原始 time 坐标，不强制规定单一日历或索引基数。
365/366 天对齐、平滑及插值属于评测层，不在配置中隐藏处理。各源的实际布局与限制
保存在对应 YAML 的 notes 中；不要将未知时间编码当成已确定规则。

## 使用与分发

通用 API 使用 `datakit.io.get_description()`、`open()`，完整示例见
[example_fuxi](../examples/dataset_io/README.md)。基础 IO 与评测不依赖 examples 模块。

安装包携带这些访问声明，安装到 `config/data_sources/` 数据目录。
`io.source_config(name)` 在源码目录或安装数据目录定位文件；也可指定 `directory=`。
外部配置始终可通过显式 YAML 路径使用，不依赖当前工作目录或名称注册表。
数据集构建目标规范独立位于 `dataset_specs/`，不与源访问配置混用，也不随基础库打包。

## 数据集构建目标规范

```text
config/
├── data_sources/    # 源数据的组织与读取声明
├── dataset_specs/   # 十份产品 YAML：目标产物应满足什么要求
└── validation/conversions/  # 源目标转换对照任务模板
```

标准使用 YAML。实际转换对照任务可在运行时提供自己的 JSON，
其中 `target_schema` 相对于该 JSON 所在目录解析。

通过 Python 函数执行标准验收：

```python
from pipelines.dataset import validate_zarr

report = validate_zarr("/data/land.zarr", "config/dataset_specs/era5_land_c8_1p5_h24.yaml")
```

output=None 不保存文件；需要保存时指定 output="/data/reports/land.json"。

陆面转换对照模板见 [land_conversion.json](validation/conversions/land_conversion.json)，
使用前需要填写实际源路径、快照标识及适用规则。转换专项入口见
[历史制作与转换示例](../examples/dataset_building/history/README.md)。
