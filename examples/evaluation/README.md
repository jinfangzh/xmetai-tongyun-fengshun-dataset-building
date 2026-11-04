# 预报数据评测与报告工作流

本目录组织不同来源预报数据的评测任务。配置指定预报、实况和可选参考数据的位置，
以及变量、起报时间、预报时效、指标参数和输出位置。运行入口加载配置并交给
`pipelines/evaluation/` 执行，指标和共享契约由 `datakit/evaluation/` 提供。

任务过程是：准备输入 → 按规则配对 → 计算与归并评分 → 写出产物 → 可选生成报告。
评测对象可以来自 AI 模型、数值预报系统或集合预报等产品。

## 准备配置并执行评分

在仓库根目录安装 `.[io,parallel,evaluation]`，按实际数据调整配置后运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
python -m pipelines.evaluation --config fdp_rmse_ens_fengqing
python -m pipelines.evaluation --config fdp_precip_single_fengqing
python -m pipelines.evaluation --config fdp_precip_ens_fengqing
```

也可以传入配置文件的完整路径。运行真实资料配置前，请按本目录中相应文件设置
数据目录和输出目录。

通用示例入口支持单个 `cfg` 和批量 `cfgs`，适用于不同预报数据的配置：

```bash
python -m examples.evaluation.run_evaluation --config weather_ts_single_tongyun
# 显式开启报告
python -m examples.evaluation.run_evaluation --config weather_rmse_single_fuxi --report
```

## 输出评分与生成报告

默认只评估并保存评分，不生成报告。传 `--report` 才会在 `<output_dir>/report/`
生成报告，标题取结果中的 run ID，也可用 `--model-name` 指定。
开启报告时会先检查整批配置的主流程是否受支持，避免运行到一半才发现模板缺失；
不支持则在任何评估开始前报错。`--no-report` 保留为兼容参数，与默认行为相同，
不能和 `--report` 同时使用。
`--forecast-root`、`--observation-root`、`--reference-root` 覆盖对应输入路径，
参考路径可以是文件。ERA5 观测路径使用 `--era5-pl-store` 和 `--era5-sfc-store`。
`--output-dir` 在单任务中直接指定输出目录，在批量任务中使用其下的 `<cfg.name>`
子目录；不指定时保留各配置的输出目录。批量任务顺序执行，遇到错误停止。

报告按 `manifest.json` 中的流程身份选择 TS、连续场或台风模板，
读取对应评分表生成统计、图表和 Markdown，不重新执行评估。
未知流程或缺少必要产物时明确报错。
生成报告需要安装 `.[evaluation,visualization]`。

| 产物主流程 | 报告 |
|---|---|
| `weather_ts_det`、`weather_ts_ens`、`fdp_precip_ts` | TS 报告 |
| `weather_field_scores` | 连续场报告 |
| `weather_typhoon_det`、`weather_typhoon_ens` | 台风报告 |

主流程是流程列表中的第一项，与 Runner 写入 manifest 的身份一致。例如
`["weather_ts_ens", "weather_ts_ens_prob"]` 可生成包含概率评分的 TS 报告；
单独的 `weather_ts_ens_prob` 没有报告模板，但仍可正常执行评估。

## 从已有评分重建报告

已有评分表和 manifest 时，可以独立生成报告，无需重跑评测：

```bash
python -m pipelines.evaluation.reporting \
    /data/evaluation/weather_ts_single_tongyun \
    --result reports/weather_ts_single_tongyun \
    --model Tongyun
```

命令支持 `--compare`、`--members` 和谱、台风报告参数。
`--result` 为相对当前工作目录的路径；
省略时默认输出到 `reports/<run_id>/`。

```python
from pathlib import Path
from pipelines.evaluation.reporting import build_report

build_report(
    Path("results/fengqing"),
    out_dir=Path("reports/fengqing"),
    model_name="FengQing",
)
```

## 组件与工作流分工

报告代码分工：

| 模块 | 职责 |
|---|---|
| `report.py` | 公开入口、参数校验、产物类型识别与派发 |
| `precipitation_plots.py` / `field_plots.py` | 组织相应类型的出图，再调用正文生成函数 |
| `ts_report.py` / `field_report.py` | 汇总结果、生成表格和 Markdown 正文，不创建图片 |
| `typhoon_report.py` / `typhoon_plots.py` | 台风场次配对、统计与正文 / 按配对统计绘图 |
| `_metadata.py` / `_plotting.py` / `radar.py` | 运行信息和区域解析 / 字体、样式与图片保存 / 雷达图及归一化 |
| `tables.py` / `views.py` / `writer.py` | 评估阶段的结果整理与写盘，供报告阶段读取 |

业务调用统一使用上面的 `build_report()` 或报告命令。内部的
`build_ts_report()` 只写正文，图片通过 `artifacts` 传入；需要完整 TS 报告时，
由 `PrecipitationPlotter.create_report()` 组织出图与正文生成。

接口约定：

| 项目 | 当前约定 |
|---|---|
| 示例脚本 | `run_fengqing_evaluation.py` 已改名为 `run_evaluation.py` |
| 示例函数 | `run_example(..., report=False)`，返回单个或批量任务的结果目录列表 |
| 示例命令 | 默认只评估；使用 `--report` 开启报告 |
| 独立报告默认位置 | `reports/<run_id>/`，相对运行目录；可显式传 `out_dir` / `--result` |
| 输入要求 | 必须有 `manifest.json` 及对应类型的结果表 |
| 返回字典 | TS、连续场使用 `artifacts["report"]`；台风使用 `artifacts["REPORT"]` |

资料路径和评估时段由配置文件指定；重建报告时直接使用已有产物目录。

相同输入下应使用相同展示名称、报告参数、绘图库版本、字体和可选依赖（如 seaborn），
以核对图片一致性。报告时间、输出路径以及台风报告的实现校验摘要会随运行变化，
不作为评分或图表差异。
