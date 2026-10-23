"""单文件日序气候态：一个 NetCDF 文件保存全年日序场。

    daily_climatology.nc
      data                      打包的气象场
      level                     打包变量标签（通道），不是气压层数值
      time                      日序或带日期的时间坐标
      其余空间坐标              保留文件原始名称和排列

read() 直接读取实际变量、维度及坐标；不预先列出气象变量。
normalize() 可按 level 标签拆分 data；365/366 天对齐、平滑和插值由评测层显式处理。
"""

from datakit.io import open_dataset
from datakit.schema.mapping import normalize_dataset

DESCRIPTION = {
    "name": "daily_climatology",
    "layout": {},
    "mapping": {"packed_variable": "data", "packed_dimension": "level"},
    "example": "daily_climatology.nc",
}


def open(path, *, selection=None):
    """配置日序气候态 NetCDF 文件的原始访问。"""
    return open_dataset(path, DESCRIPTION, selection=selection)


def normalize(dataset, variables):
    """按打包标签拆出调用方请求的变量。"""
    return normalize_dataset(dataset, variables, DESCRIPTION)
