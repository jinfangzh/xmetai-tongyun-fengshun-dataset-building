# schema

## 1. 定位与边界

`schema` 保存跨格式共享的气象维度和变量元数据。它不进行数据转换；雷达专属网格
定义放在 `radar.schema`，避免通用 schema 被领域常量污染。

数据集产品标准、生产来源映射、通道元数据生成及校验类型已移到
[`pipelines/dataset_building/definitions`](../../pipelines/dataset_building/README.md)。
ERA5 c116 与 c78 等产品使用同一归属，其常量、网格和来源映射在流程内保留。
这里暂不新增公共产品标准，保留通用维度、变量及语义模块。
后续由维护者从流程内提炼共性，再以 YAML 建立公共产品标准。

## 2. 模块与功能

- `dimensions.py`：维度别名和常用气压层集合。
- `variables.py`：变量短名、单位缩放、变量分组及分组判断函数。
- `semantics.py`：现有数据类型和时间语义枚举。

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
