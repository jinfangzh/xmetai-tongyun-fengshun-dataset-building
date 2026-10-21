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

- `tests/validation/`：标准数据集结构、质量、辅助数组及转换校验。
- `tests/dataset_building/legacy_zarr_refresh/`：旧 Zarr 更新、附属文件迁移及验证回归。
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
python -m pytest tests/validation -q -rs
python -m pytest tests/dataset_building/legacy_zarr_refresh -q -rs
```

这些测试默认使用按固定规则生成的小型 Zarr/NetCDF，实际执行数据校验、主数据更新、
通道和经纬度对齐、统计量修正、Zarr v2 写入及附属文件迁移。
合成资料测试应正常执行，并检查明确的预期结果。

`test_external_refreshed_store_contract` 使用 `integration` 标记，依赖服务器资料；
通过 `XMETAI_REFRESH_TEST_STORE` 指定已更新的 Zarr。未配置时该用例跳过。

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
