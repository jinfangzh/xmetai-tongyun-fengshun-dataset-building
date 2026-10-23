"""Diamond 站点观测：支持分日目录和平铺目录两种文件组织。

分日：                         平铺（flat=True）：
    root/                         root/
      20250101/                     2025010106.000
        06.000                      2025010112.000
        12.000

文件名为源数据的北京时间，不自动转换为 UTC。
GBK 文本有两行表头，数据列为：站号、经度、纬度、海拔、降水，后续数值列原样保留。
read() 不处理缺测哨兵；需要标准降水变量 tp 时再显式调用 normalize()。
"""

from datakit.io import open_dataset
from datakit.io.backends import station_tables
from datakit.schema.mapping import normalize_dataset

DESCRIPTION = {
    "name": "diamond_station",
    "format": {
        "encoding": "gbk",
        "header_lines": 2,
        "columns": ["station", "lon", "lat", "altitude", "precipitation"],
    },
    "layout": {"directory": "{time:%Y%m%d}", "filename": "{time:%H}.000"},
    "mapping": {
        "variables": {
            "tp": {
                "source": "precipitation",
                "unit": "mm",
                "temporal_kind": "interval_accumulation",
            }
        }
    },
    "time_zone": "Asia/Shanghai",
    "example": "20250101/06.000",
}


FLAT_DESCRIPTION = {
    "name": "diamond_station_flat",
    "layout": {"filename": "{time:%Y%m%d%H}.000"},
    "format": DESCRIPTION["format"],
    "mapping": DESCRIPTION["mapping"],
    "time_zone": "Asia/Shanghai",
    "example": "2025010106.000",
}


def open(path, *, flat=False, selection=None, layout=None):
    """配置原始站点数据访问，flat 选择平铺目录。"""
    description = FLAT_DESCRIPTION if flat else DESCRIPTION
    return open_dataset(
        path, description, selection=selection, layout=layout, backend=station_tables
    )


def normalize(dataset, variables=("tp",)):
    """显式把 precipitation 映射为标准降水变量 tp。"""
    return normalize_dataset(dataset, variables, DESCRIPTION)
