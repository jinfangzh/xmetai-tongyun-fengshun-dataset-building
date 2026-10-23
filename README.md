# xmetai-datakit

`xmetai-datakit` 是服务于 `xmetai-core` 的轻量气象数据基础库。它提供本地科学数据 I/O、xarray 数据转换、时空处理、基础统计、完整性检查、雷达解析、
可视化。仓库根目录的 `pipelines/` 将这些能力组成可复用的数据处理流程。

本项目不提供训练或推理能力，不依赖 PyTorch。下载器作为独立方案放在
`examples/dataset_downloader/`，不属于 `datakit` 公共 API。数据基础模块不包含远程对象
存储、下载服务客户端或通用工作流引擎。评测指标和数据契约位于
`datakit.evaluation`；评测运行机制位于 `pipelines/evaluation/`。

完整标准数据集的严格校验提供独立产品标准、quick/full 扫描、物理覆盖证明和发布报告。
程序测试、十套配置及服务器运行步骤见 [标准数据集校验](docs/DATASET_VALIDATION.md)。
执行示例位于 `examples/validation/`，核心库仍只接收结构化输入。

## 模块边界

```text
datakit/
├── io/             # NetCDF、GRIB、NumPy、Zarr 的显式本地读写
├── evaluation/     # 评测数据契约、指标和配对
├── schema/         # 通用维度别名和变量元数据
├── transforms/     # 坐标、变量、单位、时间和空间转换
├── statistics/     # 描述统计和气候态算法
├── validation/     # schema、完整性和数值范围检查
├── radar/          # CINRAD/MeteoNet 读取、网格定义和质量控制
└── visualization/  # 通用场、序列及雷达可视化
pipelines/
├── dataset/        # 数据集制作、检查和导出流程
└── evaluation/     # 评测流程、执行策略、结果报告和入口
examples/
├── dataset_io/          # 按数据类型命名，说明文件组织与可选变量映射
└── evaluation/configs/  # Fengqing 等具体评测配置
```

基础模块处理单一职责。`pipelines` 组合基础模块，`examples` 指定具体数据与
运行参数。数据集接入约定集中在 `examples/dataset_io/`，现有评估入口复用这些默认约定；
`datakit` 核心模块不依赖 `examples` 或 `pipelines`。CDS 与 Copernicus Marine 下载配置位于 `examples/dataset_downloader/`，
按各自 README 安装独立依赖。

`datakit.evaluation.contracts` 定义评测数据结构，`errors` 定义错误类型，
`registry` 定义可复用的组件注册表；具体组件注册与执行在
`pipelines.evaluation`。Fengqing 评测通过 `datakit.io` 读取，
通过 `pipelines.evaluation` 组织执行；
具体配置位于 `examples/evaluation/configs/`。

## Fengqing 评测

安装评测与测试依赖：

```bash
python -m pip install -e '.[all,test]'
```

Fengqing 真实资料测试位于 `tests/evaluation/integration/`。资料按测试
约定位于 `/workspace/data/shenzw/fdp_test_data/`，包含 `fengqing/`、`cra_root/`、
`cli_fake/`；站点降水测试还需要 `rain/`。在资料齐备的环境批量运行：

```bash
python -m pytest -q tests/evaluation/integration -m realdata
```

也可单独运行原有格点对表测试：

```bash
python -m pytest -q tests/evaluation/integration/test_fdp_fengqing_realdata.py -m realdata
```

缺少资料时测试会跳过。评测代码与配置可独立从 `datakit.evaluation` 导入，
评测流程由本仓库提供。

## 新用户快速开始

项目要求 Python 3.10 或更高版本，推荐使用 Python 3.11。以下命令均在仓库根目录
执行。

### 1. 创建独立环境

使用 Conda：

```bash
conda create -n xmetai-datakit python=3.11 -y
conda activate xmetai-datakit
python -m pip install --upgrade pip
```

不使用 Conda 时也可以使用标准库 `venv`：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

在 VS Code 中执行 `Python: Select Interpreter`，选择刚创建的环境，确保编辑器和
终端使用同一个 Python。

### 2. 安装

只使用内存处理、NumPy 和 Zarr 等基础能力：

```bash
python -m pip install -e .
```

需要运行全部示例和测试时，推荐一次安装完整依赖：

```bash
python -m pip install -e '.[all,test]'
```

也可以只安装需要的功能：

```bash
python -m pip install -e '.[netcdf]'         # NetCDF engines
python -m pip install -e '.[grib]'           # GRIB and ecCodes
python -m pip install -e '.[io]'             # All I/O engines
python -m pip install -e '.[interpolation]'     # SciPy interpolation
python -m pip install -e '.[parallel]'       # Dask
python -m pip install -e '.[visualization]'  # Matplotlib and Pillow
```

### 3. 验证安装

```bash
python -c "import datakit, numpy, pandas, xarray, zarr; print('datakit import OK')"
python examples/process_xarray.py
pytest -q
```

如果只安装了基础依赖，测试会跳过需要 NetCDF、SciPy 或绘图后端的用例，这是正常
现象。安装 `.[all,test]` 后应能运行完整测试。

### 4. 处理第一个文件

下面的脚本读取本地 NetCDF，统一坐标名称，然后写成 Zarr。先安装 `io` extra，
再将路径替换为自己的文件：

```python
from datakit import io
from datakit.transforms import normalize_coordinates

source = io.open("input.nc").read(load=True)
normalized = normalize_coordinates(source)
io.open("dataset.zarr").write(normalized, chunks={"time": 1})
```

打开结果并检查变量、维度和坐标：

```python
from datakit import io

dataset = io.open("dataset.zarr").read()
print(dataset)
```

### 5. 根据任务选择入口

| 目标 | 推荐入口 |
|---|---|
| 读取或写入 NetCDF、GRIB、NumPy、Zarr | `datakit.io` |
| 规范坐标、变量名、单位和归一化 | `datakit.transforms` |
| 裁剪、插值和重采样 | `datakit.transforms` |
| 计算均值、标准差、分位数和分组统计 | xarray 的归约与分组方法 |
| 检查 schema、缺测、时间连续性和数值范围 | `datakit.validation` |
| 制作可复用 Zarr 数据集产物 | `pipelines.dataset` |
| 处理 CINRAD 或 MeteoNet 雷达数据 | `datakit.radar` |

完整的可运行示例和依赖说明见 [examples/README.md](examples/README.md)。
CDS 与 Copernicus Marine 下载方案分别见
[CDS 下载说明](examples/dataset_downloader/cds_download/README_download.md) 和
[Copernicus Marine 下载说明](examples/dataset_downloader/copernicusmarine_download/README.md)。

## 基础调用

通过统一的数据访问对象读写：

```python
from datakit import io

source = io.open("input.nc").read(load=True)
io.open("dataset.zarr").write(source, chunks={"time": 1})
opened = io.open("dataset.zarr").read()
```

在内存中处理数据：

```python
from datakit.transforms import normalize_coordinates, resize

data = normalize_coordinates(source)
data = resize(data, resolution=0.25)
```

计算基础统计：

```python
mean = data.mean(dim=("time", "lat", "lon"), skipna=True)
std = data.std(dim=("time", "lat", "lon"), skipna=True)
```

## 数据集 pipeline

`pipelines.dataset` 提供路径级、产物级函数：

```python
from pipelines.dataset import convert_to_zarr

convert_to_zarr(
    "input.nc",
    "dataset.zarr",
    chunks={"time": 1, "lat": 256, "lon": 256},
)
```

## 示例

`examples/` 提供数据集接入、制作、处理、校验和评测脚本。目录和依赖说明见
[examples/README.md](examples/README.md)。

```bash
python examples/dataset_building/era5_nc_to_zarr.py --help
python examples/dataset_processing/era5_zarr_to_nc.py --help
```

运行脚本时需指定实际数据路径，并安装相应的可选 I/O 依赖。

## 命令行边界

datakit 的数据基础模块不提供统一 CLI，数据集制作的流程入口是
`pipelines.dataset`。Fengqing 评测可通过
`python -m pipelines.evaluation --config <配置名>` 运行；其他定时任务或批处理参数
仍由 `xmetai-core` 或上层应用封装。

## 与 xmetai-core 的协作

`xmetai-core` 负责模型配置、训练、推理、模型 Dataset 和业务调用时机；datakit
负责可复用的数据算法和标准数据产物。依赖方向始终为：

```text
xmetai-core  ─────>  xmetai-datakit
```

core 不应复制 NetCDF/GRIB/Zarr、基础统计和雷达解析实现；datakit 也不感知模型、
实验配置或训练框架。

## 开发验证

```bash
pytest
```

项目只支持本地文件系统路径。远程 URL、对象存储和数据下载应由上层应用或官方
数据工具负责。
