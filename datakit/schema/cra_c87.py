"""CRA c87 daily product with two grid variants."""

from __future__ import annotations

from .standard_dataset import (
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_P25,
    GRID_1P5,
    STANDARD_LAND_SEA_MASK,
)

from ._standard_products import (
    PRESSURE_CHANNELS_ASCENDING,
    DAILY_SURFACE,
    CONST_GEO,
    CONST_SMALL,
    WEIGHT_NPY,
    MASK_NPY,
    define_channels,
)

CHANNELS: tuple[str, ...] = PRESSURE_CHANNELS_ASCENDING + DAILY_SURFACE
MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant("p25.h24", GRID_P25, 24, (CONST_GEO, WEIGHT_NPY, MASK_NPY)),
    DatasetVariant("1p5.h24", GRID_1P5, 24, (CONST_SMALL, WEIGHT_NPY)),
)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="cra.c87",
    version="1.0",
    channels=define_channels(CHANNELS, radiation_seconds=86400),
    variants=VARIANTS,
    mask=MASK,
)
