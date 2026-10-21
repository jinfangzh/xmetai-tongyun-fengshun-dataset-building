# schema

## 1. 定位与边界

`schema` 保存跨格式共享的气象维度和变量元数据。它不进行数据转换；雷达专属网格
定义放在 `radar.schema`，避免通用 schema 被领域常量污染。

## 2. 模块与功能

- `dimensions.py`：维度别名和常用气压层集合。
- `variables.py`：变量短名、单位缩放、变量分组及分组判断函数。

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
