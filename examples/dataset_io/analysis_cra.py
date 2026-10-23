"""CRA 分析资料：按有效日期分目录，高空与陆面文件分开保存。

    root/
      20250101/
        ART_ATM_GLB_0P25_6HOR_ANAL_2025010100.grib2
        CRA40LAND_SURFACE_2025010100_GLB_0P25_HOUR_V1_0_0.grib
        ART_ATM_GLB_0P25_6HOR_ANAL_2025010106.grib2

文件名中的时间为分析有效时间，不是起报时间或预报时效。
原始变量由 GRIB 消息提供；不预先枚举变量，也不自动改名、换算或合并不兼容网格。
需要按 GRIB 消息取变量时，向 read() 传入相应打开参数。
评估所需的变量过滤器和单位声明位于评估层。
"""

from datakit.io import open_dataset

DESCRIPTION = {
    "name": "cra_analysis",
    "layout": {
        "directory": "{time:%Y%m%d}",
        "files": {
            "atmosphere": "ART_ATM_GLB_0P25_6HOR_ANAL_{time:%Y%m%d%H}.grib2",
            "land": "CRA40LAND_SURFACE_{time:%Y%m%d%H}_GLB_0P25_HOUR_V1_0_0.grib",
        },
    },
    "example": "20250101/ART_ATM_GLB_0P25_6HOR_ANAL_2025010100.grib2",
}


def open(path, *, selection=None, layout=None):
    """配置 CRA 原始分析文件访问。"""
    return open_dataset(path, DESCRIPTION, selection=selection, layout=layout)
