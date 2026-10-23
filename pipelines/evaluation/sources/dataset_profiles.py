"""评估变量、分组及报告单位；原始数据组织来自 examples.dataset_io。"""

from dataclasses import replace

from datakit.schema.mapping import VarSpec, load_mapping
from datakit.schema.semantics import DataKind
from examples.dataset_io import (
    analysis_cra,
    forecast_fengqing,
    forecast_fuxi,
    observation_diamond,
)
from pipelines.evaluation.sources._profile import (
    DatasetProfile,
    build_profile,
)

CRA_LAYOUT = replace(
    build_profile(analysis_cra.DESCRIPTION),
    name="cra",
    kind=DataKind("gridded_observation"),
    engine="cfgrib",
    time_kind="observation",
    variables=load_mapping(
        {
            "variables": {
                "z500": {"source": None, "unit": "m"},
                "gh": {"source": None, "unit": "m"},
                "t2m": {"source": None, "unit": "K"},
                "msl": {"source": None, "unit": "Pa"},
                "u10": {"source": None, "unit": "m/s"},
                "v10": {"source": None, "unit": "m/s"},
                "tp": {
                    "source": None,
                    "unit": "mm",
                    "temporal_kind": "interval_accumulation",
                },
            }
        }
    ).variables,
    groups={
        "z500": "ATM",
        "z": "ATM",
        "gh": "ATM",
        "t": "ATM",
        "u": "ATM",
        "v": "ATM",
        "q": "ATM",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "u10": "SURFACE",
        "v10": "SURFACE",
        "tp": "SURFACE",
    },
    grib_filters={
        "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "gh": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
        "msl": {"shortName": "msl", "typeOfLevel": "meanSea"},
        "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
        "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
        "tp": {"shortName": "tp", "stepType": "accum"},
    },
)

FENGQING_LAYOUT = replace(
    build_profile(forecast_fengqing.DESCRIPTION),
    name="fengqing",
    kind=DataKind("gridded_forecast"),
    engine="netcdf",
    time_kind="forecast",
    groups={
        "z500": "PLEVELS",
        "z": "PLEVELS",
        "gh": "PLEVELS",
        "t": "PLEVELS",
        "u": "PLEVELS",
        "v": "PLEVELS",
        "q": "PLEVELS",
        "tp": "SURFACE",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "u10": "SURFACE",
        "v10": "SURFACE",
    },
)

FENGQING_PHYS_LAYOUT = replace(
    FENGQING_LAYOUT,
    name="fengqing_phys",
    variables=load_mapping(
        {
            "variables": {
                "z500": {"source": "Z500", "unit": "m^2/s^2", "source_unit": "m^2/s^2"},
                "t700": {"source": "T700", "unit": "K"},
                "t850": {"source": "T850", "unit": "K"},
                "q700": {"source": "Q700", "unit": "g/kg", "source_unit": "g/kg"},
                "q2m": {"source": "Q2M", "unit": "g/kg", "source_unit": "g/kg"},
                "u200": {"source": "U200", "unit": "m/s"},
                "v200": {"source": "V200", "unit": "m/s"},
                "u850": {"source": "U850", "unit": "m/s"},
                "v850": {"source": "V850", "unit": "m/s"},
                "u10m": {"source": "U10", "unit": "m/s"},
                "v10m": {"source": "V10", "unit": "m/s"},
                "t2m": {"source": "T2M", "unit": "K"},
                "msl": {"source": "MSL", "unit": "Pa"},
                "tp": {
                    "source": "TP",
                    "unit": "mm",
                    "temporal_kind": "interval_accumulation",
                },
            }
        }
    ).variables,
    groups={
        "z500": "PLEVELS",
        "z": "PLEVELS",
        "gh": "PLEVELS",
        "t": "PLEVELS",
        "u": "PLEVELS",
        "v": "PLEVELS",
        "q": "PLEVELS",
        "t700": "PLEVELS",
        "t850": "PLEVELS",
        "q700": "PLEVELS",
        "u200": "PLEVELS",
        "v200": "PLEVELS",
        "u850": "PLEVELS",
        "v850": "PLEVELS",
        "tp": "SURFACE",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "q2m": "SURFACE",
        "u10m": "SURFACE",
        "v10m": "SURFACE",
    },
)

# 原生 Zarr 直接由 ERA5 读取器访问，不为它建立额外的 IO 描述。
ERA5_ZARR_LAYOUT = DatasetProfile(
    name="era5_zarr",
    kind=DataKind.GRIDDED_OBSERVATION,
    engine="zarr",
    time_kind="observation",
    variables=load_mapping(
        {
            "variables": {
                "z500": {"source": "z_500", "unit": "m^2/s^2"},
                "t700": {"source": "t_700", "unit": "K"},
                "t850": {"source": "t_850", "unit": "K"},
                "q700": {
                    "source": "q_700",
                    "unit": "g/kg",
                    "scale": 1000.0,
                    "source_unit": "kg/kg",
                },
                "u200": {"source": "u_200", "unit": "m/s"},
                "v200": {"source": "v_200", "unit": "m/s"},
                "u850": {"source": "u_850", "unit": "m/s"},
                "v850": {"source": "v_850", "unit": "m/s"},
                "msl": {"source": "msl", "unit": "Pa"},
                "t2m": {"source": "t2m", "unit": "K"},
                "d2m": {"source": "d2m", "unit": "K"},
                "q2m": {
                    "source": "q2m",
                    "unit": "g/kg",
                    "scale": 1000.0,
                    "source_unit": "kg/kg",
                },
                "u10m": {"source": "u10m", "unit": "m/s"},
                "v10m": {"source": "v10m", "unit": "m/s"},
                "ws10m": {"source": "ws10m", "unit": "m/s"},
                "tp": {
                    "source": "tp",
                    "unit": "mm",
                    "scale": 1000.0,
                    "source_unit": "m",
                    "temporal_kind": "interval_accumulation",
                },
            }
        }
    ).variables,
    groups={
        "z500": "pl",
        "t700": "pl",
        "t850": "pl",
        "q700": "pl",
        "u200": "pl",
        "v200": "pl",
        "u850": "pl",
        "v850": "pl",
        "msl": "sfc",
        "t2m": "sfc",
        "d2m": "sfc",
        "q2m": "sfc",
        "u10m": "sfc",
        "v10m": "sfc",
        "tp": "sfc",
    },
)


DEFAULT_GRIB_FILTERS = {
    name: filters for name, filters in CRA_LAYOUT.grib_filters.items() if name != "tp"
}

CLIMATOLOGY_MAPPING = load_mapping(
    {
        "variables": {
            "z500": {"source": None, "unit": "m"},
            "gh": {"source": None, "unit": "m"},
            "t2m": {"source": None, "unit": "K"},
            "msl": {"source": None, "unit": "Pa"},
            "u10": {"source": None, "unit": "m/s"},
            "v10": {"source": None, "unit": "m/s"},
        }
    }
)

DIAMOND_STATION_MAPPING = load_mapping(observation_diamond.DESCRIPTION["mapping"])


# FuXi evaluation variables and reporting units.
FUXI_LAYOUT = build_profile(forecast_fuxi.DESCRIPTION)
FUXI_ENS_LAYOUT = replace(
    build_profile(forecast_fuxi.ENSEMBLE_DESCRIPTION),
    name="fuxi_ens",
    member_dim="member",
)

# 物理量报告口径保留 z500 的位势单位；它不改变文件组织。
FUXI_PHYS_LAYOUT = replace(
    FUXI_LAYOUT,
    name="fuxi_phys",
    variables={
        **FUXI_LAYOUT.variables,
        "z500": VarSpec(source="z500", unit="m^2/s^2", source_unit="m^2/s^2"),
        "t700": VarSpec(source="t700", unit="K"),
        "t850": VarSpec(source="t850", unit="K"),
        "u200": VarSpec(source="u200", unit="m/s"),
        "v200": VarSpec(source="v200", unit="m/s"),
        "u850": VarSpec(source="u850", unit="m/s"),
        "v850": VarSpec(source="v850", unit="m/s"),
        "u10m": VarSpec(source="u10m", unit="m/s"),
        "v10m": VarSpec(source="v10m", unit="m/s"),
        "msl": VarSpec(source="msl", unit="Pa"),
    },
)
FUXI_ENS_PHYS_LAYOUT = replace(
    FUXI_PHYS_LAYOUT,
    name="fuxi_ens_phys",
    io_layout=FUXI_ENS_LAYOUT.io_layout,
    member_dim="member",
)
