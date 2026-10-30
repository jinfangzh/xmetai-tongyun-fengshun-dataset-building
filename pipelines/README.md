# 数据处理与预报数据评测工作流

`pipelines` 将数据集接入、变量处理、质量检查和指标组件组织成可复用任务。
具体数据路径、时间范围、产品标准和输出位置由任务脚本或配置提供；已有任务通常从
[examples](../examples/README.md) 中的入口开始运行。

底层格式读写复用开源库，公共数据组件位于 `datakit`，具体数据组织声明位于
[数据集接入模块](../examples/dataset_io/README.md)。这里重点组织处理步骤、输入输出和执行策略。

## 数据集工作流

`datakit.schema` 提供通用标准模型及严格 YAML 加载器；具体产品 YAML 位于
`config/dataset_specs/`，由调用方显式传入。
`pipelines.dataset` 按 production、checks、workflows、support 隔离生产定义、
验收规则、quick/full 扫描、执行配置与报告；基础库不依赖该流程包。
构建操作与单套验收、转换对照统一由 `pipelines.dataset` 提供，组合验收已移除。
详见 [数据集构建与标准校验](dataset/README.md)。

`pipelines.dataset` 提供数据集转换、提取、导出及校验任务：

| 任务 | 公开入口 |
| --- | --- |
| 组织 Zarr 产物及统计量、mask 等附属变量 | `convert_to_zarr` |
| 选择时间与通道、反归一化后导出 NetCDF | `extract_zarr_to_netcdf` |
| 按时间、通道拆分 NetCDF | `split_netcdf_by_time_and_channel` |
| 提取原始实况 | `extract_raw_truth_day` |
| 按产品标准检查数据集 | `validate_zarr` |
| 比较转换及 Zarr/NetCDF 一致性 | `validate_standard_conversion`、`validate_zarr_netcdf_consistency` |

从包入口导入所需任务，例如将工作流已准备好的 Dataset 写成产物：

```python
from pipelines.dataset import convert_to_zarr

convert_to_zarr(
    source=prepared_dataset,
    destination="/data/output/prepared.zarr",
    chunks={"time": 1},
)
```

`prepared_dataset` 由前面的处理步骤提供，调用方应先完成相应变量、坐标和单位处理。
归一化数据的导出需要匹配的均值与标准差。实际 ERA5、ORAS5 制作任务见
[数据集制作](../examples/dataset_building/README.md)，完整检查见
[数据集工作流](dataset/README.md)。

## 预报数据评测工作流

评测输入是不同来源的预报、实况及可选参考数据。流程完成按需取数、样本配对、指标状态
累积与归并，并保存评分产物；报告步骤再使用这些产物生成图表和正文。

```text
evaluation/
  sources/       数据源适配、评测变量约定和输入构造
  matching.py    时间、空间与变量匹配
  protocols.py   连续场、站点降水和台风等场景规则
  registry.py    工作流组件装配
  ensemble.py    保留评测契约的集合平均步骤
  execution/     切块、缓存、并发执行和状态归并
  reporting/     评分表、派生视图和报告
```

指标计算、共享数据契约和异常位于 `datakit.evaluation`。`sources/` 调用原始数据接入接口，
将数据适配成评测输入；原始数据集模块不依赖评测流程。专用串行入口 `serial.py` 继续保留。

## 运行任务

以下命令在仓库根目录执行。先安装相应依赖并调整配置中的数据路径和输出位置：

```bash
python -m pip install -e '.[io,parallel,evaluation]'
python -m pipelines.evaluation --list-pipelines
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
```

运行结果包括 `scores.csv`、`manifest.json` 和配置指定的派生表。
更多输入覆盖参数、批量配置和报告操作见 [评测与报告](../examples/evaluation/README.md)。
