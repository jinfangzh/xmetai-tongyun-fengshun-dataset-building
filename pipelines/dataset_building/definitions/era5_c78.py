"""ERA5 c78 six-hourly product."""

from __future__ import annotations

from .standard_dataset import (
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_P25,
    STANDARD_LAND_SEA_MASK,
)

from ._standard_products import (
    PRESSURE_CHANNELS_ASCENDING,
    CONST_OROGRAPHY,
    WEIGHT_NPY,
    MASK_NPY,
    define_channels,
)

CHANNELS: tuple[str, ...] = PRESSURE_CHANNELS_ASCENDING + (
    "t2m",
    "d2m",
    "sst",
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "msl",
    "ssr",
    "ssrd",
    "fdir",
    "ttr",
    "tp",
)
MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant("p25.h6", GRID_P25, 6, (CONST_OROGRAPHY, WEIGHT_NPY, MASK_NPY)),
)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="era5.c78",
    version="1.0",
    channels=define_channels(CHANNELS, radiation_seconds=21600),
    variants=VARIANTS,
    mask=MASK,
)
