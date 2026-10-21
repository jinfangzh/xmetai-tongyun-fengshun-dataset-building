"""Explicit product semantics shared by flat product modules.

This strict definition path does not infer unknown digit suffixes or use the
permissive channel metadata generator as validation ground truth.
"""

from .dimensions import LEVELS_13
from .standard_dataset import AuxiliaryDefinition as Aux, ChannelDefinition

# Common product constants must not depend on an individual product module.
BASE_LEVELS = tuple(reversed(LEVELS_13))
PRESSURE_VARIABLES = ("z", "t", "u", "v", "q")
PRESSURE_CHANNELS = tuple(f"{v}{p}" for v in PRESSURE_VARIABLES for p in BASE_LEVELS)
# Frozen from the server channel inventory approved on 2026-10-13.
PRESSURE_CHANNELS_ASCENDING = tuple(
    f"{v}{p}" for v in PRESSURE_VARIABLES for p in reversed(BASE_LEVELS)
)
DAILY_SURFACE = (
    "msl",
    "t2m",
    "t2m_min",
    "t2m_max",
    "d2m",
    "sst",
    "ws10m",
    "ws100m",
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "lcc",
    "mcc",
    "hcc",
    "tcc",
    "ssr",
    "ssrd",
    "fdir",
    "ttr",
    "tcw",
    "tp",
)
DEPTHS_LEGACY = (
    50,
    266,
    514,
    809,
    1177,
    1652,
    2275,
    3087,
    4118,
    5385,
    6902,
    8692,
    10803,
    13307,
    16316,
    19978,
    22114,
    27135,
    33386,
    41179,
    50863,
)
DEPTHS_CURRENT = tuple(19979 if value == 19978 else value for value in DEPTHS_LEGACY)
UNITS = {
    "z": "m2 s-2",
    "t": "K",
    "u": "m s-1",
    "v": "m s-1",
    "q": "g kg-1",
    "w": "Pa s-1",
    **{
        v: "K"
        for v in ("d2m", "t2m", "t2m_min", "t2m_max", "skt", "sst", "stl1", "stl2", "stl3", "stl4")
    },
    "q2m": "g kg-1",
    "msl": "Pa",
    "sp": "Pa",
    "sd": "m",
    "tp": "1",
    **{
        v: "m s-1"
        for v in ("u10m", "v10m", "u100m", "v100m", "ws10m", "ws100m", "ou", "ov", "uo", "vo")
    },
    **{v: "1" for v in ("lcc", "mcc", "hcc", "tcc", "cdww", "sic", "os")},
    **{v: "W m-2" for v in ("ssr", "ssrd", "fdir", "ttr")},
    "tcw": "kg m-2",
    "tcwv": "kg m-2",
    "wmb": "m",
    "mwd": "degree",
    "mwp": "s",
    "swh": "m",
    **{f"swvl{i}": "m3 m-3" for i in range(1, 5)},
    "ot": "degree_Celsius",
    "so": "1e-3",
    "to": "degrees_C",
    "sit": "m",
    "ssh": "m",
    "mlt": "m",
}


def define_channels(
    names: tuple[str, ...],
    radiation_seconds: int | None = None,
    *,
    legacy_ocean_sources: bool = False,
) -> tuple[ChannelDefinition, ...]:
    result = []
    for name in names:
        variable, level, kind, level_units = name, None, "surface", None
        for prefix in ("os", "ot", "ou", "ov", "so", "to", "uo", "vo"):
            if name.startswith(prefix) and name[len(prefix) :].isdigit():
                code = int(name[len(prefix) :])
                depths = DEPTHS_LEGACY if prefix in ("os", "ot", "ou", "ov") else DEPTHS_CURRENT
                if code not in depths:
                    raise ValueError(f"Undefined ocean depth: {name}")
                variable, level, kind, level_units = (
                    prefix,
                    code / 100,
                    (
                        "depth_below_sea_surface"
                        if prefix in ("os", "ot", "ou", "ov")
                        else "depth_below_sea"
                    ),
                    "m",
                )
                break
        if level is None:
            for prefix in ("z", "t", "u", "v", "q", "w"):
                if name.startswith(prefix) and name[len(prefix) :].isdigit():
                    code = int(name[len(prefix) :])
                    if code not in (*BASE_LEVELS, 10):
                        raise ValueError(f"Undefined pressure level: {name}")
                    variable, level, kind, level_units = prefix, code, "pressure", "hPa"
                    break
        heights = {
            "d2m": 2,
            "t2m": 2,
            "q2m": 2,
            "t2m_min": 2,
            "t2m_max": 2,
            "u10m": 10,
            "v10m": 10,
            "ws10m": 10,
            "u100m": 100,
            "v100m": 100,
            "ws100m": 100,
        }
        if name in heights:
            level, kind, level_units = heights[name], "height_above_ground", "m"
        if name in {f"{v}{i}" for v in ("swvl", "stl") for i in range(1, 5)}:
            level, kind, level_units = int(name[-1]), "soil_layer", "layer_index"
        if variable not in UNITS:
            raise ValueError(f"Undefined product variable: {name}")
        source = (
            "kg kg-1"
            if variable in ("q", "q2m")
            else (
                "J m-2"
                if variable in ("ssr", "ssrd", "fdir", "ttr")
                else "m" if variable == "tp" else UNITS[variable]
            )
        )
        variable_type = (
            "derived"
            if variable in ("ws10m", "ws100m")
            else (
                "accumulated"
                if variable in ("ssr", "ssrd", "fdir", "ttr", "tp")
                else "static" if variable == "wmb" else "instantaneous"
            )
        )
        source_names = {
            "sit": "sithick_oras",
            "sic": "siconc_oras",
            "ssh": "zos_oras",
            "mlt": "mlotst_oras",
            "so": "so_oras",
            "to": "thetao_oras",
            "uo": "uo_oras",
            "vo": "vo_oras",
            **{f"swvl{i}": f"vsw{i}" for i in range(1, 3)},
            **{f"stl{i}": f"sot{i}" for i in range(1, 3)},
            "ws10m": ("u10m", "v10m"),
            "ws100m": ("u100m", "v100m"),
        }
        source_name = (
            variable
            if legacy_ocean_sources and variable in ("sit", "sic", "ssh", "mlt")
            else source_names.get(variable, variable)
        )
        result.append(
            ChannelDefinition(
                name,
                variable,
                UNITS[variable],
                source,
                kind,
                level,
                level_units,
                radiation_seconds,
                variable_type,
                source_name,
            )
        )
    return tuple(result)


CONST_OROGRAPHY = Aux(
    "const.nc",
    "netcdf",
    "z",
    ("channel", "lat", "lon"),
    "float32",
    "const",
    ("orography", "lsm", "coslat", "sinlat", "coslon", "sinlon"),
)
CONST_GEO = Aux(
    "const.nc",
    "netcdf",
    "const",
    ("channel", "lat", "lon"),
    "float32",
    "const",
    ("geo", "lsm", "coslat", "sinlat", "coslon", "sinlon"),
)
CONST_SMALL = Aux(
    "const.nc", "netcdf", "z", ("channel", "lat", "lon"), "float32", "const", ("z", "lsm")
)
# Float32 cos(latitude)/mean(cos(latitude)) is exactly reproduced in the
# independently inspected CRA c71 and ERA5 c78/c116 weights. Pole error is
# -6.875715996557119e-8; tolerate 1e-7 without modifying stored values.
WEIGHT_NONNEGATIVE_ATOL = 1e-7
WEIGHT_NPY = Aux(
    "weight.npy",
    "npy",
    None,
    ("singleton", "lat", "singleton"),
    "float32",
    "weight",
    weight_nonnegative_atol=WEIGHT_NONNEGATIVE_ATOL,
)
MASK_NPY = Aux(
    "mask.npy",
    "npy",
    None,
    ("lat", "lon"),
    "bool",
    "external_mask",
    required=False,
    semantics_confirmed=False,
)
