# xmetai-datakit

`xmetai-datakit` 是面向气象数据处理与预报数据评测的工作流工具。它基于 xarray、Zarr 等
成熟的开源数据接口，组织数据集接入、变量标准化、数据制作、质量校验和评测流程，
减少具体任务中的重复代码，供数据处理脚本、`xmetai-core` 及其他应用复用。

评测对象是不同来源的预报数据，包括 AI 模型、数值预报系统生成的数据，以及集合预报等产品。

使用这个仓库，可以完成以下工作：

- 接入不同组织形式的气象数据集，按变量、时间、预报时效和集合成员访问数据。
- 组织变量映射、单位转换、插值、时间处理和归一化等处理步骤。
- 完成数据集制作、转换、提取、反归一化和导出。
- 按数据产品标准执行结构、时间覆盖、缺测、数值范围和转换一致性校验。
- 组织不同来源预报数据的配对、评分和结果报告，支持确定性、集合、降水和台风等场景。

`xmetai-core` 负责模型训练、推理和业务调用，`xmetai-datakit` 提供可复用的数据处理与评测流程。
工作流复用开源库的文件读写能力，并封装数据集的组织规则、处理步骤和产物约定。
当前 IO 面向本地文件系统；CDS、Copernicus Marine 下载方案
独立放在 `examples/dataset_downloader/`，按各自说明安装服务客户端依赖。

每日下载后的自动 Zarr 构建见 [Tongyun 运行手册](examples/dataset_building/TONGYUN_DAILY_OPERATIONS.md)
和 [Fengshun 运行手册](examples/dataset_building/FENGSHUN_DAILY_OPERATIONS.md)。

## 代码结构

围绕这些任务，仓库分为基础组件、可复用工作流和具体应用三个部分：

```text
xmetai-datakit-server/
├── datakit/                    工作流使用的基础组件
│   ├── io/                     数据访问、文件定位与多文件组合
│   ├── schema/                 变量、维度、单位和数据产品约定
│   ├── transforms/             坐标处理、插值、单位转换和归一化
│   ├── validation/             数据结构、完整性和质量检查
│   ├── evaluation/             评测指标、共享数据契约和异常
│   ├── radar/                  雷达数据读取与质量控制
│   ├── visualization/          绘图和动画
│   └── runtime/                日志和运行资源工具
├── pipelines/                  可复用的数据处理与评测工作流
│   ├── dataset/                数据集构建、转换、导出、验收和报告
│   └── evaluation/             数据适配、配对、评测执行和结果输出
├── examples/                   具体数据和任务的使用入口
│   ├── dataset_io/             YAML 驱动的统一读取与导出示例
│   ├── dataset_building/       标准数据集构建策略、维护入口及历史脚本
│   ├── dataset_processing/     数据处理与导出脚本
│   ├── dataset_downloader/     数据下载方案
│   ├── validation/             基础 NetCDF/Zarr 对照示例
│   └── evaluation/             评测配置和运行脚本
├── config/data_sources/        独立数据源 YAML 声明
├── tests/                      单元测试与集成测试
└── docs/                       数据标准、校验规范和运行说明
```

`datakit` 提供变量约定、数据转换、质量检查和指标计算等基础组件，IO 层衔接开源数据接口。
`pipelines` 将这些组件组织成完整任务，例如制作数据集或完成一次预报数据评测。
`examples` 保存完整调用示例和实际运行脚本，指定数据路径、处理范围与输出位置。
使用现有工作流时，通常从对应的脚本或配置开始。

依赖方向是 `examples → pipelines / datakit`、`pipelines → datakit`。
源数据访问声明独立位于 `config/data_sources/`；基础库和评测不导入示例模块。

### 数据集接入

数据接入遵循一个原则：**先读取原始数据，再显式进行标准化。**

FuXi、Fengqing、CRA、Diamond 等具有特定文件组织的数据，入口位于
`config/data_sources/`。每份 YAML 描述目录组织、文件内容和时间含义，并声明
必要的读取规则，让工作流能够按时间、变量、预报时效等条件定位和取得数据。
不同数据集保留自己的差异，不要求套用相同的参数模板。

底层 NetCDF、GRIB、NumPy、Zarr 读写复用成熟开源接口。标准文件可以直接使用 xarray、
Zarr 或 NumPy；需要按数据集规则发现、筛选和组合文件时，通过 YAML 声明调用通用 IO。

变量映射可以省略，也可以只描述部分变量。映射说明源变量与标准变量的对应关系，
不是读取白名单。原始读取保留源变量名和单位，显式调用 `apply_dataset_mapping()` 时才执行映射和换算。

单位换算、对数变换和归一化的顺序由具体任务明确安排。制作与反归一化导出应使用匹配的
通道、统计量和处理约定；产品结构与校验标准见 [`config/dataset_specs`](config/dataset_specs)。

相关说明见 [通用 IO](datakit/io/README.md) 和 [数据集接入](examples/dataset_io/README.md)。

### 评测流程

评测将数据接入、样本配对、指标计算和结果输出分开：

```text
预报、实况、参考数据
        ↓
数据源适配：按评测请求取得所需数据
        ↓
匹配与场景规则：对齐时间、空间和变量，生成评测样本
        ↓
指标计算：累积可合并状态，计算评分
        ↓
结果输出：scores.csv、manifest.json、配置指定的派生表
        ↓
可选报告：读取评分产物，生成图表和正文
```

执行层负责切块、数据缓存、并发执行和状态归并。指标计算与执行策略分离，
报告生成读取已有产物，无需重新运行评测。

| 任务或职责 | 查看位置 |
| --- | --- |
| 数据集转换、提取与导出 | `pipelines/dataset/` |
| 独立生产定义、构建产物验收与报告 | `pipelines/dataset/` |
| 具体数据集制作与处理任务 | `examples/dataset_building/`、`examples/dataset_processing/` |
| 评测配置与运行入口 | `examples/evaluation/` |
| 某个数据集怎么组织、怎么读 | `config/data_sources/` |
| 数据接入的完整调用示例 | `examples/dataset_io/example_fuxi.py` |
| 数据怎么转换、插值或归一化 | `datakit/transforms/` |
| 数据产品应满足什么标准 | `datakit/schema/` |
| 数据检查算法 | `datakit/validation/` |
| 评测数据怎么接入 | `pipelines/evaluation/sources/` |
| 预报和实况怎么配对 | `pipelines/evaluation/matching.py` |
| 降水、台风等场景规则 | `pipelines/evaluation/protocols.py` |
| 指标计算公式 | `datakit/evaluation/metrics/` |
| 切块、缓存、并发和归并 | `pipelines/evaluation/execution/` |
| 评分表和报告生成 | `pipelines/evaluation/reporting/` |

## 使用方式

项目要求 Python 3.10 或以上版本。以下命令均在仓库根目录执行，示例中的 `/data/` 路径
需要替换为实际数据路径。

### 1. 安装环境

建议先创建独立 Python 环境，例如：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

只使用基础能力：

```bash
python -m pip install -e .
```

需要常见文件后端、多文件读取和评测能力：

```bash
python -m pip install -e '.[io,parallel,evaluation]'
```

绘图和测试依赖可另外安装，也可以一次安装完整依赖：

```bash
python -m pip install -e '.[visualization,test]'
# 或：python -m pip install -e '.[all,test]'
```

验证安装：

```bash
python -c "import datakit; print('datakit import OK')"
```

### 2. 接入数据集并准备工作流输入

以 FuXi 集合预报为例：

```python
from datakit import io
from datakit.schema.mapping import apply_dataset_mapping, load_mapping

description = io.get_description("config/data_sources/forecast_fuxi_ensemble.yaml")
source = io.open(
    "/data/fuxi",
    description,
    selection={
        "init_time": ["2025-01-01"],
        "lead_time": [6, 12],
    },
)

records = source.discover()  # 只定位文件，不读取数据正文。

with source.read() as raw:
    standard = apply_dataset_mapping(raw, load_mapping(description), ["z500", "t2m"])
    result = standard.load()
```

接入其他数据集时，先查看对应模块开头的组织说明。结果为 xarray Dataset，
可继续交给处理、校验或评测步骤。多文件数组读取默认使用 Dask，需要安装 `parallel` extra。
直接使用通用 IO 写出时，默认拒绝覆盖已有输出，确需覆盖应显式指定 `mode="w"`。

### 3. 制作和导出数据集

数据集的创建、时间追加、验收和比较直接使用 `pipelines.dataset` 的 Python API，
调用方式见 [流程说明](pipelines/dataset/README.md)。
ERA5/ORAS5 制作流程见 [数据集制作示例](examples/dataset_building/history/README.md)。

历史制作脚本会删除已有目标目录后重新生成数据，运行前应确认 `--destination`。

对已经归一化的 Zarr，可以选择时间和通道，反归一化后导出：

```bash
python -m examples.dataset_processing.era5_zarr_to_nc \
  --zarr-path /data/era5.zarr \
  --save-path /data/output/era5_20250101.nc \
  --start-date 2025-01-01 \
  --time-interval 1 \
  --channels t2m,z500
```

均值和标准差默认从输入数据中读取，也可通过 `--stats-path` 指定外部目录。
有通道标签的统计量按标签对齐；无标签数组按位置对应，长度必须匹配。
更多制作与处理入口见 [examples](examples/README.md)。

### 4. 检查数据集

标准数据集通过独立 YAML 描述产品结构、覆盖范围和校验要求，再执行检查。
十份产品验收标准位于 `config/dataset_specs/`，运行配置、证据绑定和报告由
`pipelines/dataset/` 组织，基础库不依赖该流程包：

```python
from pipelines.dataset import validate_zarr

report = validate_zarr("/data/era5.zarr", "config/dataset_specs/era5_c116_p25_h6.yaml")
# 需要保存时传 output="outputs/validation/era5.json"。
```

YAML 标准必须与实际数据产品对应。默认完整扫描，抽样检查可指定 mode="quick"。
output=None 不保存文件；指定报告文件时，路径须位于输入数据和附件目录之外。

产品标准见 [`config/dataset_specs`](config/dataset_specs)，操作接口见 [数据集工作流](pipelines/dataset/README.md)。

### 5. 运行预报数据评测

先查看可用流程：

```bash
python -m pipelines.evaluation --list-pipelines
```

在 `examples/evaluation/configs/` 中选择配置，调整预报、实况和参考数据路径，以及评测时间、
时效和输出目录，然后运行：

```bash
python -m pipelines.evaluation --config fdp_rmse_single_fengqing
```

也可以直接指定配置文件：

```bash
python -m pipelines.evaluation \
  --config examples/evaluation/configs/fdp_precip_ens_fengqing.py
```

输出以 `scores.csv` 和 `manifest.json` 为基础，其他评分表由配置决定。
支持的流程还可通过示例入口生成报告：

```bash
python -m examples.evaluation.run_evaluation \
  --config weather_rmse_single_fuxi \
  --report
```

评测参数、报告支持范围和已有结果的报告生成方式见
[评测配置与报告](examples/evaluation/README.md)。

### 6. 验证代码修改

先运行 IO、数据适配及评测指标的回归测试：

```bash
python -m pytest \
  tests/test_package_import.py \
  tests/test_dataset_conversion.py \
  tests/test_dataset_io.py \
  tests/test_evaluation_source_adaptation.py \
  tests/test_evaluation_transforms.py \
  tests/evaluation/unit \
  -q
```

真实资料测试位于 `tests/evaluation/integration/`，准备对应资料后运行：

```bash
python -m pytest tests/evaluation/integration -m realdata -q
```

资料位置与覆盖范围按各测试及其 fixture 约定，缺少资料时相应用例会跳过。
构建、转换和完整数据集校验测试位于 `tests/dataset_building/`。运行全仓库测试使用 `python -m pytest`。

CDS 与 Copernicus Marine 的下载方式分别见
[CDS 下载说明](examples/dataset_downloader/cds_download/README.md) 和
[Copernicus Marine 下载说明](examples/dataset_downloader/copernicusmarine_download/README.md)。
