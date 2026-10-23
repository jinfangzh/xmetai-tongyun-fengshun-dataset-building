"""风清集合预报：同一起报和时效的高空、地面文件分开存放。

    root/
      20250101/
        Fengqing_1.0_GLB_PLEVELS_OP25_6HOR_ENS_FCST_2025010100_006.nc
        Fengqing_1.0_GLB_SURFACE_OP25_6HOR_ENS_FCST_2025010100_006.nc
        Fengqing_1.0_GLB_SURFACE_OP25_6HOR_ENS_FCST_2025010100_012.nc

文件名包含起报时间和预报小时数；006 是 6 小时，不是文件序号。
PLEVELS 与 SURFACE 在同一起报、时效下合并；成员来自文件内部的 member 维度。
level、time、dtime 仅在明确为单例记录维度时压缩；不猜测或插值空间坐标。
read() 保留源变量和单位，下面的 mapping 只由 normalize() 显式应用。
"""

from datakit.io import open_dataset
from datakit.schema.mapping import normalize_dataset

DESCRIPTION = {
    "name": "fengqing_ensemble",
    "layout": {
        "directory": "{init_time:%Y%m%d}",
        "filename": "Fengqing_1.0_GLB_{group}_OP25_6HOR_ENS_FCST_{init_time:%Y%m%d%H}_{lead_time:03d}.nc",
        "groups": ["PLEVELS", "SURFACE"],
    },
    "mapping": {
        "variables": {
            "z500": {
                "source": "Z500",
                "unit": "m",
                "scale": 1 / 9.80665,
                "source_unit": "m^2/s^2",
            },
            "tp": {
                "source": "TP",
                "unit": "mm",
                "temporal_kind": "interval_accumulation",
            },
            "t2m": {"source": "T2M", "unit": "K"},
            "msl": {"source": "MSL", "unit": "Pa"},
            "u10": {"source": "U10", "unit": "m/s"},
            "v10": {"source": "V10", "unit": "m/s"},
        },
        "squeeze_dims": ["level", "time", "dtime"],
        "transpose_dims": ["member", "lat", "lon"],
        "member_dim": "member",
        "ensure_member_dim": True,
    },
    "lead_time": {"units": "hours"},
    "structure": {"record_dimensions": ["level", "time", "dtime"]},
    "example": "20250101/Fengqing_1.0_GLB_SURFACE_OP25_6HOR_ENS_FCST_2025010100_006.nc",
}


def open(path, *, selection=None, profile=None, layout=None):
    """配置原始风清文件访问。"""
    return open_dataset(
        path, DESCRIPTION, selection=selection, profile=profile, layout=layout
    )


def normalize(dataset, variables, *, profile=None):
    """显式应用变量映射和单位换算。"""
    return normalize_dataset(dataset, variables, DESCRIPTION, profile=profile)
