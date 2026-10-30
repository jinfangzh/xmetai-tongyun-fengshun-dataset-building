# 校验工作流的规范与基准证据

本目录保存校验工作流采用的产品规则、独立基准和证据解释。制作任务输出数据集后，
校验任务按这些规则检查结构、覆盖与数值，并将实际检查结果写入本次运行报告。
这里的基准用于解释和约束检查，不是从待检数据自动生成的期望值。

当前校验条款见 [标准数据集校验计划](VALIDATION_PLAN.md)，运行、依赖及报告阅读方法见
[程序使用说明](../DATASET_VALIDATION.md)。实例配置位于
[`config/dataset_specs`](../../config/dataset_specs)，每套独立声明产品、模板版本和批次。

十份 YAML 是运行时标准的唯一维护来源，严格加载后由通用程序执行。
旧 JSON profiles、inventory 和纯验收 Python 产品模块已删除；c116 来源映射、生产常量及
通道元数据生成代码保留在流程内。版本、根身份和 YAML SHA256 分别绑定，不能相互替代。
固定标准模型和严格 YAML 加载位于 `datakit.schema`；检查、运行配置、扫描和回执 IO 位于
`pipelines.dataset` 的操作、配置、报告和证据模块；来源映射与旧元数据生成位于 `old/raw_truth.py`、`old/channel_metadata.py`，
数据集验收直接调用 validate_zarr。基础库的原有检查不迁移。生产/转换回执及 `snapshot_id` 由独立 JSON 的
`execution` 绑定，不写回 YAML；运行 JSON 不是另一套产品标准。
本流程不再包含 core 探测、消费者或模型测试回执核验；验收结论仅覆盖声明的数据检查。
跨数据集组合检查已删除；每次调用 validate_zarr 只验收一套，批量循环由外围实现。
`datakit/schema/specifications.py` 定义固定结构、约束和推导，`datakit/schema/parser.py` 严格加载 YAML。
`configuration.py` 定义运行规格、调用基础加载器并装配执行输入；旧转换规格位于 `old/conversion_validation.py`。
旧目录层级已移除，基础检查位于 datakit.validation，pipeline 按操作、证据、配置和报告平铺。
checks 按格式、数值、存储和转换组织规则，转换回执也由转换规则核验。
quick/full 扫描及资源控制位于 `validate.py`；`reports.py` 提供任务报告与结论汇总，validate_zarr 按需保存单份 JSON。

| 文件 | 用途 |
|---|---|
| `VALIDATION_PLAN.md` | 本版已确认条款、十套产品实例和 quick/full 完成条件 |
| [SCHEMA_YAML_DESIGN.md](SCHEMA_YAML_DESIGN.md) | 单套 YAML 的通用结构目录、字段职责、变量关联及维护约定 |
| [SCHEMA_YAML_MERGE_DESIGN.md](SCHEMA_YAML_MERGE_DESIGN.md) | 未采用的历史合并方案，仅供追溯，不是运行时标准 |
| `PRODUCT_BASELINE_2026-10-13.json` | 独立冻结的现有通道顺序、纬度端点和权重盘点；产品回归的对照依据 |
| `IDENTITY_BASELINE_2026-10-13.json` | 独立冻结的目录与根 ID 映射及当次元数据摘要；实例显式映射的依据 |
| `WEIGHT_PRECISION_2026-10-13.md` | 本批 float32 权重误差复现及非负容差依据 |

基准是历史只读盘点，不是本次扫描报告，也不能替代新批次的生产完成证明。
新批次追加时间或调整身份时更新实例配置；通道、网格、掩膜或处理声明发生变化时重新
选择或制定产品版本。不得把被检数据的值自动写成期望标准。

运行报告、临时环境及退出当前范围的历史校准材料不纳入版本控制。当前 YAML 不依赖
历史有效区域 NPY 或暂定质量阈值；相应诊断不参与本版验收。
