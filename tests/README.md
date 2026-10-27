# 测试指南

测试使用本仓库的 `datakit`、`pipelines` 源码。评测用例、资料路径和检验口径见
[evaluation/integration/README.md](evaluation/integration/README.md)。

## 安装与运行

Python 3.10 或更高版本，在仓库根目录运行：

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q -rs
```

GRIB 等可选格式的真实资料测试需要完整依赖：

```bash
python -m pip install -e '.[all,test]'
```

按范围运行：

```bash
# 不依赖外部资料的测试
python -m pytest tests -m 'not realdata and not integration' -q -rs
# 评测单元测试
python -m pytest tests/evaluation/unit -q -rs
# 合成资料的串行、多进程和 resume 集成回归
python -m pytest tests/evaluation/integration/test_weather_field_execution.py -v
# 真实资料对表
python -m pytest tests/evaluation/integration -m realdata -v -rs
# 单个用例
python -m pytest tests/evaluation/unit/test_weather_field_rmse.py::test_rmse_known_values_matches_hand_calculation -v
```

`realdata` 是筛选标记，不会默认禁用测试。缺资料和可选后端的检查在 fixture
或运行函数中执行，用例仍会被收集，并逐项显示跳过原因。跳过不等于数值通过。
7 项已知指标契约缺陷保留明确的 skip 原因，修复对应实现后应恢复执行。

## 组织与约定

- `tests/dataset_building/samples/`：集中维护旧 Zarr 和标准 v3 小样本工厂、手工预期结果及独立测试 YAML；不保存生产资料。
- `tests/dataset_building/conftest.py`：共享样本准备，每项测试在自己的临时目录中生成资料。
- `tests/dataset_building/test_conversion.py`：从旧 Zarr 到更新产物和附件的实际流程测试。
- `tests/dataset_building/test_validation.py`：单套、独立多套校验和转换回执衔接的实际流程测试。
- `tests/dataset_building/acceptance/`：构建产物标准、质量、附件、转换、证据和模块边界回归。
- `tests/dataset_building/legacy_zarr_refresh/`：单文件示例 `examples/dataset_building/legacy_zarr_refresh.py` 的更新、附属文件迁移、CLI 和断点续写回归。
- `tests/evaluation/unit/`：指标、配对、协议和执行配置的单元测试。
- `tests/evaluation/integration/`：正式评测入口的合成回归与真实资料对表。
- `tests/evaluation/helpers/`：共享资料工厂、批次构造、路径检查和独立参考基准。
- `tests/test_era5_nc_to_zarr.py`、`tests/test_oras5_nc_to_zarr_88.py`：数据制作回归。
- `tests/test_evaluation_source_adaptation.py`、`tests/test_evaluation_transforms.py`：IO、适配与变换的一致性回归。

测试文件命名为 `test_*.py`，函数命名为 `test_<行为>`。优先按准备输入、调用接口、
断言结果组织代码；共享 fixture 放在最近的 `conftest.py`，仅在确有多个调用者时
抽取 helper。fixture 的作用域应明确，公共输入不应被测试原地修改。

文件产物写入 `tmp_path` 或 `tmp_path_factory`。pytest 提供这些 fixture；不依赖
仓库固定输出目录、历史结果或用例执行顺序。Fengqing 测试保留手动调试入口，
其固定输出目录仅用于直接调试，不用于 pytest。

断言应校验关键数值、样本数、状态和产物语义。数组使用 NumPy 或 xarray 的
比较工具；浮点比较应注明容差来源。筛选表格后先确认行数或非空，再取第一行、
遍历或调用 `.all()`，避免空输出误通过。异常测试尽可能限定异常类型及关键报错信息。

手算预期值、独立脚本结果与框架回归基线应说明来源。数值基准独立于生产指标计算；
共享基准集中维护，不为消除失败随意放宽容差。修复代码前记录已知缺陷的原因，
避免将普通失败改成跳过。

## 标准数据校验和 Zarr 更新

```bash
# 样本驱动的主要流程测试
python -m pytest tests/dataset_building/test_conversion.py tests/dataset_building/test_validation.py -q -rs
# 同时运行保留的细节回归
python -m pytest tests/dataset_building -q -rs
# 只运行原有细节回归
python -m pytest tests/dataset_building/acceptance -q -rs
python -m pytest tests/dataset_building/legacy_zarr_refresh -q -rs
```

主要流程测试按“固定样本 → 实际程序 → 已知结果”组织，不另设 integration 目录：

- 旧 Zarr：2 个时次、2 个通道、2×2 网格。分别准备通道/纬度需要重排和已经对齐的输入，
  输出都与 `samples/legacy_expected.json` 中手工确定的结果比较。核对串行/双线程转换、
  mean/std 修正、逐值结果、v3 原生维度和斜线块键、掩膜、常量、权重和附件迁移；中断后验证只续写未完成时次。
- 标准 v3：12 个时次、3×4 网格。第一套为 `t2m/q1000/tp` 三通道，第二套为
  `q1000/t2m` 两通道，时间错开 10 天，各自使用独立的标准和运行配置。测试 quick 的
  3 块选择与 full 的 12 块扫描、逐套结论、JSON/CSV/Markdown 报告和块台账。
- 故障副本：通道错序、NaN、超界值、无效 std、缺块、损坏压缩块。核对具体失败检查项，
  同批的另一套正常资料仍应通过。另验证 quick 未选中的坏块只能由 full 发现。
- 转换证据：独立构造通道重排、纬度翻转和 2/4/8 倍线性映射的 v3 参考源，
  运行源目标对照、生成真实回执，再接入 full 校验；错误数值不能形成有效验收结论。
  refresh 的 v3 产物由另一组实际转换测试直接验证，两者不互相替代。

样本定义和预期值集中保存，实际 Zarr/NetCDF/NPY、运行配置、回执和报告均写入 pytest
临时目录；测试核对转换源、校验输入及标准没有被程序修改，不依赖服务器资料或网络下载。
基础小样本 YAML 独立编写；另有十套明确标记为 miniature 的产品布局回归，从正式 YAML 保留完整通道、层次和处理语义，仅缩小测试时间及网格，并对照独立服务器通道基准。它们不能冒充满足完整生产时空范围的资料。
原有细节回归保留并复用集中样本，不因增加流程测试删减已有断言。

**格式边界：** 更新器读取旧 v2 或 v3 输入，按显式引用的 YAML 写出实际 Zarr v3。
十套产品的 miniature 产物分别运行 quick 和 full：quick 应通过，full 应完成主块读取及存储覆盖检查；没有独立生产冻结证明时仍不标记发布通过。负例保留“content_version 写 v3、底层实际 v2”的拒绝检查，并验证全 NaN 帧真实落块、删除块后失败。
样本回归通过仅证明所选软件行为正确，服务器全量资料仍应通过正式校验入口另行验收。

这些测试默认使用按固定规则生成的小型 Zarr/NetCDF，实际执行数据校验、主数据更新、
通道和经纬度对齐、统计量修正、Zarr v3 写入及附属文件迁移。
合成资料测试应正常执行，并检查明确的预期结果。
数据验收不包含 core/消费者/模型回执检查。回归覆盖旧字段拒绝、JSON/CSV/Markdown
报告栏目和无模型依赖时的生产入口导入；原有 `core-70` 通道子集不属于删除范围。
组合工作流及其接口已删除。回归验证两套时间范围不相交的数据仍可按各自标准批量验收，
不会加入时间交集、共享网格或掩膜相同的跨数据集门槛。
生产、规则、流程和支撑分别位于 dataset_building 下的 production、checks、workflows、support，
源码中的 acceptance 层已移除；测试目录仍以 acceptance 表示验收业务。
规则按格式、数值、存储和转换分组；附件布局和数值分别归入 format、values。
quick/full 及批量执行位于 workflows，报告位于 support，转换回执业务核验位于 checks/conversion。
模块边界回归检查新导入路径及旧实现没有重复残留，报告回归覆盖完成结论、输出形状及退出码。
固定标准模型和严格 YAML 解析位于 `datakit.schema.datasets`，运行绑定留在流程层。
边界回归独立导入并加载全部十份打包标准，禁止反向加载流程或生产，检查标准子包没有直接依赖数据读取后端。

`test_external_refreshed_store_contract` 使用 `integration` 标记，依赖服务器资料；
通过 `XMETAI_REFRESH_TEST_STORE` 指定已更新的 Zarr。未配置时该用例跳过。
它只检查 v2 基本结构，不代替转换数值对照或严格 YAML 验收。

## 代码风格

`tests/ruff.toml` 统一测试代码的导入顺序、100 字符行宽及基础错误检查，
并检查未使用导入和循环闭包等问题。安装 Ruff 后执行：

```bash
python -m pip install ruff
python -m ruff check tests
python -m ruff format --check tests
```

修正格式可运行 `python -m ruff format tests`。格式检查不能代替测试执行，
真实资料测试的跳过情况也应随结果一并说明。
