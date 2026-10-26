# schema

## 1. 定位与边界

`schema` 保存跨格式共享的气象维度、变量元数据及独立数据集 YAML 声明。它不进行数据转换；雷达专属网格
定义放在 `radar.schema`，避免通用 schema 被领域常量污染。

生产来源映射和通道元数据生成位于
[`pipelines/dataset_building/production`](../../pipelines/dataset_building/production)。
ERA5 c116 的生产常量、网格和来源映射在流程内保留，不作为独立验收真值。
[`datasets/`](datasets/) 是数据集标准子包，保存固定标准模型、严格 YAML 解析器和十份独立 YAML，
替代旧 JSON profile 和纯验收 Python 产品声明，并随安装包分发。
运行快照、回执绑定及执行 JSON 位于
[`pipelines/dataset_building`](../../pipelines/dataset_building) 的 support，检查和扫描分别位于
checks、workflows。本层不反向导入流程、生产定义或数据读取后端。

## 2. 模块与功能

- `dimensions.py`：维度别名和常用气压层集合。
- `variables.py`：变量短名、单位缩放、变量分组及分组判断函数。
- `semantics.py`：现有数据类型和时间语义枚举。
- `datasets/specifications.py`：数据集、网格、通道、数组、掩膜、附件和静态质量规则的结构、约束及期望值推导。
- `datasets/inputs.py`：独立 YAML 严格解析、路径解析、原始字节摘要及标准文件变更检查；不读取待检数据或执行 JSON。
- `datasets/*.yaml`：每套数据的完整、独立预期标准；不使用公共配置继承或合并注册表。

## 3. 公共函数导入

```python
from datakit.schema import (
    DIMENSION_ALIASES,
    LEVELS_13,
    SHORT_NAME_MAPPING,
    is_accumulated,
)
```

## 4. 核心调用示例

```python
from datakit.schema import DIMENSION_ALIASES, LEVELS_13

canonical = DIMENSION_ALIASES.get("latitude", "latitude")
pressure_levels = LEVELS_13
```

数据集标准可以脱离执行流程独立加载：

```python
from datakit.schema.datasets import load_schema

schema, locations = load_schema("datakit/schema/datasets/era5_land_c8_1p5_h24.yaml")
```

`pipelines.dataset_building.support.inputs` 中的 `load_validation_schema` 在此基础上构造运行实例，
`load_validation_run` 另外装配独立执行 JSON。
固定目标模型位于 `datasets/specifications.py`，`support/specifications.py` 只定义运行绑定和转换对照规格。
数组、数据属性、掩膜取值、通道元数据、网格起点/方向及数据版本由 YAML 显式提供，
不再提供 `standard_arrays`、`DATA_ATTRIBUTES` 或 `STANDARD_LAND_SEA_MASK` 作为第二份默认标准。
直接构造模型时也必须显式提供这些声明；推荐从 YAML 加载，再通过 `dataclasses.replace` 派生实例。
`build_standard_channel_info` 需要显式传入标准中的 `channel_metadata`，不自行指定恢复公式。
模型仍检查扫描器支持的数组布局、类型和整帧分块；加载器仍拒绝不支持的归一化语义和公式，
这些是执行能力边界，不会用于补齐缺失声明，也不表示支持任意 YAML 布局或表达式。
原 acceptance 层不再保留。运行回执和 `snapshot_id`
不能写入 YAML；固定标准不因添加回执而改变摘要。
