# xmetai-core 数据能力覆盖说明

## 结论

当前 datakit 以“满足 core 数据使用的最基础能力闭环”为边界，不再按
`xmetai-core/xmetai/utils` 的历史文件结构逐项搬运。core 中与数据相关的实现原本
混合了通用数据算法、模型 Dataset、训练张量操作和一次性脚本；datakit 只覆盖前两者
中可独立于训练框架的部分。

## 已覆盖能力

| core 数据需求 | datakit 位置 |
|---|---|
| NetCDF、GRIB、NPY/NPZ 本地读写 | `io` |
| Zarr 读写、追加、合并和坐标更新 | `io.zarr` |
| 坐标、维度和变量名规范化 | `schema`、`transforms` |
| 归一化、反归一化和单位缩放 | `transforms` |
| 时序筛选、累计量转增量和重采样 | `transforms.temporal` |
| 样本序列窗口索引 | `datasets.sequences` |
| 空间裁剪、插值和分辨率调整 | `transforms.spatial` |
| mean/std、分位数、月日分位数和时次气候态 | `statistics` |
| 差分统计产物 | `pipelines.dataset.build_difference_statistics` |
| 评估网格的余弦纬度权重 | `datakit.evaluation.matching` |
| 维度、坐标、完整性、NaN/Inf 和范围检查 | `validation` |
| CINRAD/MeteoNet 读取和雷达质量控制 | `radar` |
| NetCDF/GRIB/NumPy 转 Zarr | `pipelines.dataset` |
| Zarr 统计产物、检查和 NetCDF 导出 | `pipelines.dataset` |
| 雷达和集合数据集转 Zarr | `pipelines.dataset` |
| 普通气象场和雷达可视化 | `visualization` |

## 明确不覆盖

- PyTorch Tensor 运算及 torch 依赖守卫。
- 模型 Dataset、Sampler、DataLoader 和训练批处理逻辑。
- 训练框架内部的分布式评估和实验汇总；离线气象指标由 `datakit.evaluation` 提供。
- 数据下载、官网 API 包装和远程对象存储。
- CLI、Registry、配置加载、工作流引擎和任务调度。
- 仅为单个历史脚本服务、core 当前没有稳定调用需求的工具函数。

## 两个仓库的边界

```text
xmetai-core
├── 模型 Dataset 和数据配置
├── 训练、推理与评估
└── 调用 datakit pipeline
          │
          ▼
xmetai-datakit
├── 本地格式 IO
├── xarray 数据算法
├── 基础统计、验证和雷达处理
└── 标准数据集制作
```

依赖必须保持单向：datakit 不导入 core。core 中仍存在的历史数据脚本可以逐步变成
datakit pipeline 的薄调用层，但不应把 argparse、日志、进度条或训练框架依赖反向
移入 datakit。
