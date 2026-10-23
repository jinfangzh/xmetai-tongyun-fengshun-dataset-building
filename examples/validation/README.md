# 数据集校验工作流

本目录将独立产品标准、待检数据路径和扫描参数组织成校验任务，用于检查制作或迁移后的
数据集，并输出检查结果、扫描完成状态和验收证据。profile 描述期望结构与覆盖范围，
`datakit.validation` 执行检查，`pipelines.dataset` 组织扫描和报告。

输入是产品 profile 与数据集；输出是输入目录之外的报告。报告区分抽样检查通过、
完整扫描完成和发布条件满足，使用时应分别查看对应字段。

## 运行任务

标准数据集校验的完整说明、配置审核、程序测试和服务器命令见
[标准数据集校验](../../docs/DATASET_VALIDATION.md)。在仓库根目录使用 `python -m` 运行：

```bash
python -m examples.validation.validate_standard_dataset \
  --inventory examples/validation/profiles/inventory.json \
  --mode quick --output /path/outside/data/reports
```

## 选择标准与阅读结果

十份 profile 独立对应十套数据，使用 confirmed-format 范围，仅纳入已确认规则。
完整分类条款和逐套标准见 [校验计划](../../docs/validation/VALIDATION_PLAN.md)。
quick_passed 判断预定抽样；quick 中 scan_complete/release_passed 始终 false。
转换、组合和真实 core 探测是独立专项入口，不参与本版十套 quick 判定。
旧 `validate_netcdf.py` / `validate_zarr_nc.py` 保留原有用途。

## 验证任务实现

程序测试按职责放在 `tests/validation/`，在仓库根目录执行：

```bash
python -m pytest tests/validation -q
```

配置中的产品标准和批次覆盖独立冻结；需要追加时间时新建或更新对应 profile。
基准证据说明见 [校验规范与基准证据](../../docs/validation/README.md)。
