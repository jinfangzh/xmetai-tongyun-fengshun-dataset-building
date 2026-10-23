"""CRA 月日时次气候态：一个文件对应每年重复的一个月、日、小时。

    root/
      ART_ATM_GLB_0P25_CLI_ANAL_010100.grib2
      ART_ATM_GLB_0P25_CLI_ANAL_010106.grib2
      ART_ATM_GLB_0P25_CLI_ANAL_010200.grib2

010106 表示 1 月 1 日 06 时，不包含年份；读取后保留 cycle_time 标签。
文件中实际存在的变量由 GRIB 提供，不声明固定变量清单。
其他后缀或文件名通过 layout 参数明确指定，不凭文件名伪造观测年份。
"""

from datakit.io import open_dataset

DESCRIPTION = {
    "name": "cra_climatology",
    "layout": {"filename": "ART_ATM_GLB_0P25_CLI_ANAL_{cycle_time:%m%d%H}.grib2"},
    "cycle": {"period": "year"},
    "example": "ART_ATM_GLB_0P25_CLI_ANAL_010100.grib2",
}


def open(path, *, selection=None, layout=None):
    """配置原始月日时次气候态访问。"""
    return open_dataset(path, DESCRIPTION, selection=selection, layout=layout)
