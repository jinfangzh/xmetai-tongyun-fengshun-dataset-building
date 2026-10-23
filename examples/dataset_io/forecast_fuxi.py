"""伏羲数据的文件组织。

单模型预报：
    root/
      20250101/                 起报日期，约定为当天 00 时
        001.nc                  起报后 6 小时
        002.nc                  起报后 12 小时
      20250102/
        001.nc

集合预报：
    root/
      20250101/
        member_0/               成员 0
          001.nc
          002.nc
        member_1/               成员 1
          001.nc
          002.nc

一个 NetCDF 文件对应一个预报时效；默认时效 = 文件编号 × 6 小时。
起报时间、预报时效和集合成员从路径获取，NC 内部变量名与单位在读取时保留。
下面的 mapping 仅在显式调用 normalize() 时应用。
"""

# mapping 的键为标准变量名，source 为源变量名；只在 normalize() 时应用。
from datakit.io import open_dataset
from datakit.schema.mapping import normalize_dataset

DESCRIPTION = {
    "name": "fuxi",
    "layout": {
        "directory": "{init_time:%Y%m%d}",
        "filename": "{lead_index:03d}.nc",
    },
    "lead_time": {"hours_per_index": 6},
    "mapping": {
        "case_insensitive": True,
        "variables": {
            "z500": {
                "source": "z500",
                "unit": "m",
                "scale": 1 / 9.80665,
                "source_unit": "m^2/s^2",
            },
            "tp": {
                "source": "tp",
                "unit": "mm",
                "temporal_kind": "interval_accumulation",
            },
            "t2m": {"source": "t2m", "unit": "K"},
            "q700": {"source": "q700", "unit": "g/kg", "source_unit": "g/kg"},
            "q2m": {"source": "q2m", "unit": "g/kg", "source_unit": "g/kg"},
        },
    },
    "example": "20250101/001.nc",
}

# 集合预报的成员来自目录；变量映射与普通预报相同。
ENSEMBLE_DESCRIPTION = {
    "name": "fuxi_ensemble",
    "layout": {
        "directory": "{init_time:%Y%m%d}/member_{member:d}",
        "filename": "{lead_index:03d}.nc",
    },
    "lead_time": {"hours_per_index": 6},
    "mapping": DESCRIPTION["mapping"],
    "example": "20250101/member_0/001.nc",
}


def open(path, *, ensemble=False, selection=None, profile=None, layout=None):
    """配置原始数据访问，不改名、不换算单位。"""
    description = ENSEMBLE_DESCRIPTION if ensemble else DESCRIPTION
    return open_dataset(
        path, description, selection=selection, profile=profile, layout=layout
    )


def normalize(dataset, variables, *, profile=None):
    """显式应用变量映射和单位换算。"""
    return normalize_dataset(dataset, variables, DESCRIPTION, profile=profile)
