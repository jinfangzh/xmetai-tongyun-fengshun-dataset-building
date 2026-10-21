# Fengqing 评测配置

本目录存放具体数据位置、评测时间和输出位置。通用评测流程位于
`pipelines/evaluation/`，指标与数据契约位于 `datakit/evaluation/`。

在仓库根目录安装 `.[evaluation]` 后，可运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
python -m pipelines.evaluation --config fdp_rmse_ens_fengqing
python -m pipelines.evaluation --config fdp_precip_single_fengqing
python -m pipelines.evaluation --config fdp_precip_ens_fengqing
```

也可以传入配置文件的完整路径。运行真实资料配置前，请按本目录中相应文件设置
数据目录和输出目录。

`run_fengqing_evaluation.py` 在评测落盘后默认生成基础结果报告，保存为
`<output_dir>/report/REPORT.md` 和同目录 PNG 图。需要安装
`.[evaluation,visualization]`；只输出评测表格可传 `--no-report`。
基础报告按评分维度和起报展示已保存的结果，不包含旧项目的多模型分析或谱诊断。

已有 CSV 时，可以独立生成报告，无需重跑评测：

```python
from pathlib import Path
from pipelines.evaluation.reporting import build_report

build_report(Path("results/fengqing"), model_name="FengQing")
```
