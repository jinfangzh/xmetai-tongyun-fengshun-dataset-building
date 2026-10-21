"""Distinct ocean naming/semantic versions; no automatic alias conversion."""

from __future__ import annotations

from dataclasses import replace

from .standard_dataset import (
    AuxiliaryDefinition as Aux,
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_P25,
    GRID_1P5,
    STANDARD_LAND_SEA_MASK,
)

from ._standard_products import (
    DEPTHS_LEGACY,
    DEPTHS_CURRENT,
    CONST_GEO,
    CONST_SMALL,
    define_channels,
    WEIGHT_NONNEGATIVE_ATOL,
)

LEGACY_CHANNELS: tuple[str, ...] = tuple(
    f"{v}{d}" for v in ("os", "ot", "ou", "ov") for d in DEPTHS_LEGACY
) + (
    "sit",
    "ssh",
    "sic",
    "mlt",
)
CURRENT_CHANNELS: tuple[str, ...] = tuple(
    f"{v}{d}" for v in ("so", "to", "uo", "vo") for d in DEPTHS_CURRENT
) + (
    "sit",
    "ssh",
    "sic",
    "mlt",
)
CHANNELS: tuple[str, ...] = LEGACY_CHANNELS
MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
SMALL_AUXILIARIES = (
    CONST_SMALL,
    Aux(
        "weight.nc",
        "netcdf",
        None,
        ("lat",),
        "float32",
        "weight",
        weight_nonnegative_atol=WEIGHT_NONNEGATIVE_ATOL,
    ),
    Aux(
        "land_mask.nc",
        "netcdf",
        "mask",
        ("lat", "lon"),
        "bool",
        "land_copy",
        required=False,
        equals_mask="land_mask",
    ),
    Aux(
        "mask.nc",
        "netcdf",
        "lsm",
        ("lat", "lon"),
        "float32",
        "external_mask",
        required=False,
        semantics_confirmed=False,
    ),
    Aux(
        "sst_mask.nc",
        "netcdf",
        "sst",
        ("lat", "lon"),
        "float32",
        "sst_mask",
        required=False,
        semantics_confirmed=False,
    ),
)
VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant("1p5.h24", GRID_1P5, 24, SMALL_AUXILIARIES),
    DatasetVariant(
        "p25.h24",
        GRID_P25,
        24,
        (
            CONST_GEO,
            Aux(
                "weight.nc",
                "netcdf",
                None,
                ("lat",),
                "float32",
                "weight",
                weight_nonnegative_atol=WEIGHT_NONNEGATIVE_ATOL,
            ),
        ),
    ),
)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="oras5.c88",
    version="1.0",
    channels=tuple(
        (
            replace(channel, unconfirmed_fields=("units", "source_units"))
            if channel.variable == "os"
            else channel
        )
        for channel in define_channels(CHANNELS, legacy_ocean_sources=True)
    ),
    variants=VARIANTS,
    mask=MASK,
    semantic_review=("Legacy ocean salinity physical scale requires independent confirmation",),
)
CURRENT_PRODUCT: DatasetProduct = DatasetProduct(
    product_id="oras5.c88",
    version="2.0",
    channels=tuple(
        (
            replace(channel, unconfirmed_fields=("units", "source_units"))
            if channel.variable == "so"
            else channel
        )
        for channel in define_channels(CURRENT_CHANNELS)
    ),
    variants=VARIANTS,
    mask=MASK,
    semantic_review=(
        "Current production layout and salinity scale require independent release confirmation",
    ),
)
