# 数据集 IO 示例

[example_fuxi.py](example_fuxi.py) 展示按 YAML 描述读取 FuXi 集合预报、选择源变量并导出 NetCDF。

修改 `main()` 中的输入、输出路径、变量名和筛选条件后，在仓库根目录运行：

```bash
python -m examples.dataset_io.example_fuxi
```

默认读取成员 0 的 6、12 小时预报，对应文件为：

```text
/data/fuxi/20250101/member_0/001.nc
/data/fuxi/20250101/member_0/002.nc
```

读取保留源变量名、单位和缺测值，不执行变量映射或数值换算。
数据在源关闭之前写出，拒绝覆盖已有输出；可以直接在 IDE 中进入 `main()` 调试。
多文件读取需要安装 `parallel` extra，例如 `pip install -e '.[io,parallel]'`。

数据描述位于 [config/data_sources](../../config/data_sources/)。
`io.source_config(name)` 定位项目内的 YAML，也可直接向 `io.open()` 传入自定义 YAML 路径。
单文件访问使用 `io.open(path)`。接口说明见 [通用 IO](../../datakit/io/README.md)，
声明规则见 [配置说明](../../config/README.md)。
