# 评测测试

包含单元测试和集成测试，调用本仓库的 `pipelines.evaluation` 和
`datakit.evaluation`。

## 单元测试

单元测试位于 `tests/evaluation/unit/`，共 233 个参数化展开后的用例。

| 测试模块 | 验证内容 |
| --- | --- |
| `test_ensemble_metrics.py` | CRPS、SpreadError 和状态合并 |
| `test_execution_profiles.py` | 资源画像和执行策略 |
| `test_precipitation_categorical.py` | 列联表和分类指标 |
| `test_precipitation_diag.py` | 观测跨度和降水累积 |
| `test_precipitation_io.py` | Diamond 发现、解码、站点白名单 |
| `test_precipitation_probability.py` | 集合概率评分 |
| `test_precipitation_spatial.py` | 邻域比例和 FSS |
| `test_spec_periods.py` | 单时段、多时段解析和起报日发现 |
| `test_temporal_window.py` | 观测时间窗口、站级缺测和错误诊断 |
| `test_typhoon_diag.py` | 报文解析、中心诊断、配对和误差 |
| `test_typhoon_io_writer.py` | BABJ Reader 和台风 CSV/元数据输出 |
| `test_typhoon_metrics.py` | 确定性及集合台风误差 |
| `test_typhoon_seed.py` | 种子时刻选择 |
| `test_weather_field_acc.py` | 距平相关系数 |
| `test_weather_field_activity.py` | 活跃度 |
| `test_weather_field_bias.py` | 平均偏差 |
| `test_weather_field_grid_matcher.py` | 经纬度、网格和时间配对 |
| `test_weather_field_rmse.py` | 均方根误差 |
| `test_weather_field_spectrum.py` | 二维谱和纬向谱 |
| `test_weather_field_spherical_bands.py` | 球谐谱和波段能量 |

7 个上游已知缺陷用例保留原有 skip 及原因：指标状态合并校验、概率指标资源画像、
BSS 权重一致性和 FSS 无有效样本状态。它们不计入验证通过数。

## 集成测试

| 测试文件 | 验证内容 |
| --- | --- |
| `test_fdp_fengqing_realdata.py` | Fengqing × CRA40：确定性、集合误差、活跃度与谱对表 |
| `test_fdp_precip_station_realdata.py` | Fengqing × Diamond：分类列联表及集合概率评分对表 |
| `test_precip_ts_realdata.py` | FGVP：24 小时站点降水分类检验 |
| `test_rainstorm_ts_realdata.py` | FuXi / FGVP 控制成员：暴雨过程逐日和整过程列联表对表 |
| `test_precip_ens_realdata.py` | FuXi 集合：24 小时 TS 与 6 小时概率评分 |
| `test_typhoon_realdata.py` | BABJ 种子选择、确定性与集合台风归档对表 |
| `test_weather_field_realdata.py` | FuXi 确定性 × ERA5：连续场指标对表 |
| `test_weather_field_ens_realdata.py` | FuXi 集合 × ERA5：集合及连续场指标对表 |
| `test_weather_field_execution.py` | 合成天气场：进程执行与串行一致、resume 结果复现 |

另外保留 `test_fengqing_serial_realdata.py`，验证专用串行入口。

## 运行

在仓库根目录安装依赖并运行：

```bash
python -m pip install -e '.[evaluation,test]'
python -m pytest tests/evaluation -v
# FDP 真实资料链还需要 GRIB 解码等可选依赖：
python -m pip install -e '.[all,test]'
```

不依赖外部资料的单元测试及执行回归：

```bash
python -m pytest tests/evaluation/unit -v
python -m pytest tests/evaluation/integration/test_weather_field_execution.py -v
```

只运行真实资料测试：

```bash
python -m pytest tests/evaluation/integration -m realdata -v -rs
```

`realdata` 是 pytest 标记，并不会默认禁用测试；资料缺失时用例会说明原因并跳过。
跳过不代表已经验证了真实资料上的数值结果。上游基准值及其来源说明保留在各文件中，
其中部分是框架回归基线，尚未完成独立参考脚本复核。

## 配置与资料

测试配置由 `pipelines.evaluation.config.load_config()` 从
`examples/evaluation/configs/` 加载。`weather_*` 配置声明数据源、
指标及执行设置；`weather_ts_single_fgvp` 的集成测试使用单日 fixture。
多时段 `periods` 支持：配置与声明透传，按各段候选起报日的并集发现预报，
不探测空档日期。`periods` 与 `start_date/end_date` 同时给出会报配置错误。
测试会覆盖实际起报范围、输出位置和执行模式。

暴雨测试另配 `rainstorm_ts_single_fuxi`、`rainstorm_ts_single_fgvp_ctrl` 两份配置，
过程编目位于 `examples/evaluation/configs/rainstorm_cases_2025.py`。
测试从配置中选取 202521 过程的 9 段；资料根为
`${XMETAI_FIXTURE_DIR:-/workspace/data/shenzw}/rainstorm_test_data`。

| 资料组 | 默认位置及覆盖方式 |
| --- | --- |
| FDP | `${XMETAI_FIXTURE_DIR:-/workspace/data/shenzw}/fdp_test_data`；同时覆盖配置中的预报、实况与气候态路径 |
| 天气场 | `${XMETAI_FIXTURE_DIR:-/workspace/data/shenzw}/weather_field_test_data` |
| 站点降水 | `${XMETAI_FIXTURE_DIR:-/workspace/data/shenzw}/precip_test_data` |
| 台风 | `/workspace/data/shenzw/typhoon_test_data` |

天气场可分别用 `FUXI_OUTPUT`、`FUXI_ENS_OUTPUT`、`ERA5_PL_STORE`、`ERA5_CLIMO`
覆盖路径。降水可用 `FGVP_OUTPUT`、`FUXI_ENS_OUTPUT`、`STATION_OBS`、`STATION_LIST`、
`BSS_REF` 覆盖路径。不同测试可能使用同名环境变量指向不同资料，通常优先使用统一的
`XMETAI_FIXTURE_DIR`，或逐文件设置环境后运行。

台风路径使用 `XMETAI_BABJ_DIR`、`XMETAI_FUXI_DET_ROOT`、`XMETAI_FUXI_ENS_ROOT`、
`XMETAI_TC_DET_ARCHIVE`、`XMETAI_TC_ENS_ARCHIVE`。

公共 fixtures 位于 `tests/evaluation/conftest.py`，台风、降水和天气场辅助工具分别为
`helpers/typhoon.py`、`helpers/precipitation.py`、`helpers/weather_field.py`。
ACC、活跃度测试共用 `helpers/batches.py` 的批次构造。
通用和专用串行 Fengqing 入口共用 `helpers/fengqing_reference.py` 的独立基准及对表检查。
外部资料检查由 `helpers/paths.py` 和集成目录的 fixtures 统一处理，缺资料时逐项跳过，
不会在收集阶段跳过整个模块。所有 pytest 输出保存在临时目录。

串行与多进程比较会检查非空结果、成功状态及数值一致性。
resume 回归还检查块状态文件存在，并拦截第二次计算以确认复用确实发生。
