# 从 xmetai-core 提炼数据能力

本文记录当前稳定边界，不提供旧 datakit API 的兼容别名。

## 迁入 datakit

| core 中的通用逻辑 | datakit 入口 |
|---|---|
| NetCDF、GRIB、NumPy、Zarr 本地读写 | `datakit.io` |
| 坐标、维度、变量、单位和归一化 | `datakit.transforms` |
| 时间、空间和通道操作 | `datakit.transforms` |
| mean/std、分位数和气候态 | `datakit.statistics` |
| 差分统计产物 | `pipelines.dataset` |
| 数值范围、无效值、时间和 schema 检查 | `datakit.validation` |
| CINRAD/MeteoNet 读取和反射率质控 | `datakit.radar` |
| 数据集转换、统计产物、雷达和集合 Zarr 制作 | `pipelines.dataset` |
| 普通气象场和雷达图像 | `datakit.visualization` |

## 保留在 core

- PyTorch Dataset、DataLoader、Sampler 和 Tensor 运算。
- 模型输入窗口、训练增强、推理批次和 GraphCast 等模型结构。
- 训练日志、checkpoint、loss、模型注册和分布式评估。
- CLI 参数解析、项目配置、进度展示和业务调度。

## 当前 canonical API

| 历史职责 | 当前入口 |
|---|---|
| 自动格式 dispatch | 已删除，使用 `io.read_netcdf/read_grib/read_numpy/read_zarr` |
| `resize/crop` | `datakit.transforms` |
| 时间重采样 | `datakit.transforms` |
| 雷达 schema 和 reader | `datakit.radar` |
| Zarr 原子读写 | `datakit.io` |
| 格式转 Zarr、Zarr 检查和导出 | `pipelines.dataset` |
| 统计量计算 | `datakit.statistics` |
| 统计量文件构建 | `pipelines.dataset` |

依赖方向为：

```text
core 或 examples -> pipelines -> datakit 基础模块
```

datakit 不导入 core，也不探测或预埋训练框架守卫。
