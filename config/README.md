# 数据源描述配置

`data_sources/` 只保存 YAML。每份声明独立描述一个数据集及其文件组织方式，
不导入 Python、不绑定机器路径，也不组织评测或验收。读取选择、根目录与输出位置由调用方提供。

| YAML | 文件组织 |
| --- | --- |
| `analysis_cra.yaml` | CRA 分日分析，高空和陆面分文件 |
| `climatology_cra.yaml` | CRA 月日时次 GRIB 气候态，无观测年份 |
| `climatology_daily.yaml` | 单文件日序气候态，data 按 level 标签打包 |
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
- `format.name`：数据格式，可为 `netcdf`、`grib`、`zarr`、`numpy`、
  `diamond_station`、`station_probability` 或 `babj`。
- `format` 的其他字段：文本编码、表头和列定义。
- `lead_time`、`cycle`、`time_zone`：时效编号换算、周期和源时区。
- `structure`、`member`：记录维度及成员组织。
- `thresholds`：站点气候概率表各列对应的阈值。
- `mapping`：变量名称、单位及线性换算、坐标别名、打包字段和维度规则。

`format.name` 描述数据格式，IO 层据此选择库内后端；不接受 Python 导入路径或
可执行表达式，未知格式会报错。旧 `reader` 字段不再接受。声明的容器格式参与实际读取与写出；
必要时调用方可显式通过 `read(format=...)` 覆盖，例如使用 NetCDF 小样本验证 GRIB 目录布局。
缺测值和源时间默认原样保留。标准化由调用方显式执行，不在 read() 中隐式发生。
新的文件格式需要先实现 IO 后端；已有格式的新布局可直接增加 YAML。

日序气候态保留原始 time 坐标；当前声明允许日期或日序，未规定单一日历或索引基数。
365/366 天对齐、平滑及插值属于评测层，不在配置中隐藏处理。各源的实际布局与限制
保存在对应 YAML 的 notes 中；不要将未知时间编码当成已确定规则。

## 使用与分发

通用 API 使用 `datakit.io.get_description()`、`open_dataset()` 和
`datakit.schema.mapping.normalize_dataset()`，完整示例见
[io_example](../examples/dataset_io/README.md)。基础 IO 与评测不依赖 examples 模块。

安装包携带这些访问声明，安装到 `config/data_sources/` 数据目录。
`io.source_config(name)` 在源码目录或安装数据目录定位文件；也可指定 `directory=`。
外部配置始终可通过显式 YAML 路径使用，不依赖当前工作目录或名称注册表。
产品发布验收标准仍由具体构建任务维护，不与源访问配置混用。
