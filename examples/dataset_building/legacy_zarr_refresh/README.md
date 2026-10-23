# 旧 Zarr 数据集整理工作流

本目录将已有归一化 Zarr 的盘点、通道与坐标对齐、统计量修正、写出和校验组织成独立任务。
输入是旧数据集和 JSON 配置，输出是按配置整理的新 Zarr 及检查记录，必要时迁移
`const`、`weight` 和 `mask` 等附属文件。

该任务保持为独立示例，尚未纳入 `datakit` 公共 API。它处理已经归一化的数据，
与 `era5_nc_to_zarr.py` 的原始 NetCDF 制作任务分开，不推断新的物理单位换算。
JSON 配置保存实际路径和数据集专属规则，执行组件负责源数据检查、对齐、Zarr v2 写出和数值验证。

## 盘点输入与试运行

在仓库根目录执行。先复制 `config.example.json` 并替换实际路径，再查看盘点结果与计划：

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json inspect --dataset all

python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json pilot --dataset oras5 \
  --output-root /path/to/output --count 4 --dry-run
```

先写出小段并检查，再运行完整整理：

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json pilot --dataset oras5 \
  --output-root /path/to/output --count 4 --full-data-check

python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json convert --dataset oras5 \
  --output-root /path/to/output --workers 4
```

## 独立校验输出

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json validate --dataset oras5 \
  --zarr /path/to/output/oras5.refreshed.zarr --full-data-check
```

## 迁移附属文件

目标 Zarr 完成后再迁移附属文件，先查看计划：

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json auxiliary --dataset oras5 \
  --output-root /path/to/output --target-name oras5.refreshed.zarr --dry-run
```

默认保留 Zarr 内嵌 mask。确认两套掩膜语义一致后，才能使用 `--refresh-embedded-mask`
从 `const.nc/lsm` 替换它。`const.nc` 保留原有通道与顺序，不补造额外通道。

## 验证工作流

```bash
python -m pytest tests/dataset_building/legacy_zarr_refresh -q
```

默认测试在 pytest 临时目录生成确定性的小型 Zarr 和 NetCDF，执行完整写读流程，覆盖通道
重排、纬度反向、统计量修正、掩膜、Zarr v2 点分隔 chunk 键、校验记录与附属文件对齐。
生成值保持简单，以便直接断言预期结果。

默认测试不依赖服务器数据，也不向 Git 提交数据 fixture。外部 Store 契约检查默认跳过；
设置 `XMETAI_REFRESH_TEST_STORE` 指向已完成的服务器产物后，可显式运行对应集成测试。
服务器路径缺失不能跳过生成数据的端到端和附属量测试，这些测试必须在本地和 CI 中通过。

读取服务器固定长度字符串数组需要 Zarr 3.1.6 或更高版本。
