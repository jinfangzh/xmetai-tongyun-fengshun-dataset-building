"""根据 description.implementation 选择共享读写后端。

后端只实现 read_records(records, description) 与 write(dataset, root, description)。
文件发现统一由 数据访问对象 负责，不在每个后端重复包装。
布局名称保持稳定，文件名描述实际职责。
files.py 负责单文件访问，file_collection.py 负责描述驱动的多文件访问。
"""

from importlib import import_module

_MODULES = {
    "forecast_files": "file_collection",
    "time_files": "file_collection",
    "variable_time_files": "file_collection",
    "grouped_time_files": "file_collection",
    "climatology_files": "file_collection",
    "xarray_dataset": "files",
    "diamond_station": "station_tables",
    "station_probability": "station_tables",
    "babj": "babj",
}


def implementation(name):
    """返回实现模块；不同路径组织可以共用同一个多文件后端。"""
    if name not in _MODULES:
        raise ValueError(f"Unknown dataset implementation: {name!r}")
    return import_module(f"{__name__}.{_MODULES[name]}")
