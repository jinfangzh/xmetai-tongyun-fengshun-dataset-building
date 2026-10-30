# Copernicus Marine 数据获取工作流

本目录负责 ORAS5 等海洋数据的源数据获取：根据 Profile 和 Batch 声明组织日度请求、
文件落盘、重试和续传，实际下载复用官方 Copernicus Marine Toolbox。

输入是数据集、变量、空间范围、日期范围和输出根目录；输出是逐日 NetCDF、计划清单及
状态日志。当前 ORAS5 Profile 每天下载 8 个变量，保存在同一个 NetCDF 中，
可继续交给 `examples/dataset_building/history/oras5_nc_to_zarr_88.py` 制作数据集。
本任务独立于 CDS 下载方案，使用自己的服务客户端依赖和认证配置。

## 配置分层

- `profiles/*.yaml`：数据集 ID、变量、经纬度范围、深度范围和文件标签。
- `batch_task.yaml`：Profile 组合、输出根目录、手工日期范围、并发和重试。
- 日期使用首尾均包含的 `YYYY-MM-DD`，不在 Profile 中填写。

示例输出：

```text
copernicusmarine_data/
├── oras5_daily/
│   └── 1993/
│       ├── 19930101_oras5.nc
│       └── 19930102_oras5.nc
└── _records/batch_task/
    ├── plan.txt
    └── status.jsonl
```

下载先写成同目录下的 `<日期>_oras5.part.nc`，成功后原子替换为最终文件。只有完成日志与目标文件同时存在才会跳过任务。默认一次只运行一个下载，避免给远端服务施加过多并发请求。

## 安装与认证

在仓库根目录执行：

```bash
python -m pip install -r examples/dataset_downloader/copernicusmarine_download/requirements.txt
copernicusmarine login
```

用户名和密码由 Copernicus Marine Toolbox 管理，不写入源码或 YAML。

## 执行

在仓库根目录运行：

```bash
# 检查生成的请求与目标，不连接远端
python examples/dataset_downloader/copernicusmarine_download/copernicusmarine_download.py \
  examples/dataset_downloader/copernicusmarine_download/batch_task.yaml --dry-run

# 只生成目标计划
python examples/dataset_downloader/copernicusmarine_download/copernicusmarine_download.py \
  examples/dataset_downloader/copernicusmarine_download/batch_task.yaml --plan-only

# 正式下载；默认根据 status.jsonl 续传
python examples/dataset_downloader/copernicusmarine_download/copernicusmarine_download.py \
  examples/dataset_downloader/copernicusmarine_download/batch_task.yaml

# 忽略完成记录重新下载
python examples/dataset_downloader/copernicusmarine_download/copernicusmarine_download.py \
  examples/dataset_downloader/copernicusmarine_download/batch_task.yaml --force-redownload
```

下载器调用官方 `copernicusmarine.subset()`，并显式传入 `output_directory` 和 `output_filename`。Profile 默认不固定 `dataset_version`、`dataset_part` 和 `service`，由 Toolbox 从当前目录选择；需要完全复现实验时，可在先核对目录后取消 Profile 中相应字段的注释。
