# 校验规范与基准证据

当前校验条款见 [标准数据集校验计划](VALIDATION_PLAN.md)，运行、依赖及报告阅读方法见
[程序使用说明](../DATASET_VALIDATION.md)。实例配置位于
[`examples/dataset_building/configs/profiles`](../../examples/dataset_building/configs/profiles)，每套独立选择产品和批次。

流程内每个产品定义均提供 `CHANNELS`、`MASK`、`VARIANTS`、`PRODUCT`，类型及声明顺序一致。
完整 `PRODUCT` 在对应产品文件中用具名参数声明；产品汇总文件只负责导入和选择。
已有生产常量名称在流程内保留，c116 的来源映射和元数据生成代码也位于流程内；
旧公共 schema 导入路径撤出。多版本产品另行显式声明版本，不改变默认选择。

| 文件 | 用途 |
|---|---|
| `VALIDATION_PLAN.md` | 本版已确认条款、十套产品实例和 quick/full 完成条件 |
| `PRODUCT_BASELINE_2026-10-13.json` | 独立冻结的现有通道顺序、纬度端点和权重盘点；产品回归的对照依据 |
| `IDENTITY_BASELINE_2026-10-13.json` | 独立冻结的目录与根 ID 映射及当次元数据摘要；实例显式映射的依据 |
| `WEIGHT_PRECISION_2026-10-13.md` | 本批 float32 权重误差复现及非负容差依据 |

基准是历史只读盘点，不是本次扫描报告，也不能替代新批次的生产完成证明。
新批次追加时间或调整身份时更新实例配置；通道、网格、掩膜或处理声明发生变化时重新
选择或制定产品版本。不得把被检数据的值自动写成期望标准。

运行报告、临时环境及退出当前范围的历史校准材料不纳入版本控制。当前 profile 不依赖
历史有效区域 NPY 或暂定质量阈值；相应诊断不参与本版验收。
