"""CRA c71 six-hourly product."""

from __future__ import annotations

from .standard_dataset import (
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_CRA_C71_P25,
    STANDARD_LAND_SEA_MASK,
)

from ._standard_products import (
    PRESSURE_CHANNELS_ASCENDING,
    CONST_OROGRAPHY,
    WEIGHT_NPY,
    define_channels,
)

CHANNELS: tuple[str, ...] = PRESSURE_CHANNELS_ASCENDING + (
    "t2m",
    "q2m",
    "u10m",
    "v10m",
    "msl",
    "tp",
)
MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant("p25.h6", GRID_CRA_C71_P25, 6, (CONST_OROGRAPHY, WEIGHT_NPY)),
)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="cra.c71",
    version="1.0",
    channels=define_channels(CHANNELS),
    variants=VARIANTS,
    mask=MASK,
)
