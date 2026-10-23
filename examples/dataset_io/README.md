# 数据集接入

每个文件开头先说明实际目录组织、文件内容和时间含义，再声明必要的读取规则。
同一数据集的组织声明、原始读写和必要的变量映射放在同一个文件里。
评测请求、样本配对和评测输入契约由 `pipelines/evaluation` 负责。
文件名使用数据类型前缀；不同数据集保留自己的细节，不要求套用相同的声明模板或函数参数。
通用读写和转换复用 `datakit`，核心库不依赖本目录。

| 文件 | 组织形式与主要区别 |
| --- | --- |
| `forecast_fuxi.py` | 按起报日期、时效编号存 NetCDF；集合成员来自目录 |
| `forecast_fengqing.py` | 高空/地面分文件；时效直接写在文件名中，成员来自文件内部 |
| `analysis_cra.py` | 按分析有效日期归档，高空和陆面 GRIB 分文件 |
| `observation_diamond.py` | 分日或平铺的站点文本，文件名为北京时间 |
| `observation_babj.py` | 一个台风一份报文，原始读取保留全部时效与额外列 |
| `climatology_cra.py` | 按月、日、小时保存 GRIB，不补年份 |
| `climatology_daily.py` | 单文件全年日序场，变量可打包在 data/level 中 |
| `climatology_station_probability.py` | 按月日时次保存站点降水概率表，阈值列有固定含义 |

需要多种实际目录组织时，直接保留清楚的声明，例如 FuXi 的 `DESCRIPTION` /
`ENSEMBLE_DESCRIPTION`、Diamond 的 `DESCRIPTION` / `FLAT_DESCRIPTION`。
文件内不维护 `DESCRIPTIONS` 注册表，不为评估单位口径派生新的数据集描述。
声明不写 `implementation`；特殊文本解析器由各文件的 `open()` 显式选择。

```python
from examples.dataset_io import forecast_fuxi, observation_diamond

source = forecast_fuxi.open(
    "/data/fuxi",
    ensemble=True,
    selection={"init_time": ["2025-01-01"], "lead_time": [6, 12]},
)
with source.read() as raw:
    result = raw.load()

# 只有显式调用才执行变量改名和单位换算。
standard = forecast_fuxi.normalize(result, ["z500", "t2m"])

with observation_diamond.open("/data/station", flat=True).read(all_files=True) as raw:
    rain = observation_diamond.normalize(raw, ["tp"]).load()
```

`mapping` 可以省略，也可以只声明部分变量；它不是读取白名单。
读取默认包含文件实际提供的变量；调用方可在读取后选择所需字段。
键为标准变量名，`source` 为源变量名；换算关系为 `标准值 = 源值 * scale + offset`。
CRA、BABJ 和站点气候概率表不需要额外映射，也不提供多余的 `normalize()` 接口。
文本的编码、表头和列顺序是解析所需的格式声明，与变量映射不同。

`open()` 返回访问对象，使用其 `read()` / `discover()` / `write()`。
Zarr 数据直接使用 `zarr.open()` 或 `xarray.open_zarr()`，不为每个数据集再包 IO。
ERA5 的评测取数适配位于 `pipelines/evaluation/sources/era5_zarr.py`。
写回原始格式使用原始数据，不自动逆转映射。

评估角色、GRIB 变量过滤器、文件分组路由和物理量报告口径位于
`pipelines/evaluation/sources/dataset_profiles.py`，包括 FuXi 的报告单位约定。
BABJ 轨迹组装、气候态平滑插值和站点概率查询位于 `pipelines/evaluation/sources/`，
由评测组件调用。既有评估配置名称保持可用。

公共访问和声明处理位于 `datakit.io`：`open_dataset()` 配置原始访问，
`get_description(DESCRIPTION)` / `get_layout(DESCRIPTION)` 返回独立副本。
调用方直接选择数据集声明，不通过示例目录中的名称注册表查找。
变量映射执行位于 `datakit.schema.mapping.normalize_dataset()`；
评估数据源约定类型与构造函数位于 `pipelines/evaluation/sources/_profile.py`。
`dataset_io/__init__.py` 只保留包说明，本目录没有公共 `_profile.py`。
数据集转换、导出和校验流程位于 `pipelines.dataset`，评估执行位于 `pipelines.evaluation`。
