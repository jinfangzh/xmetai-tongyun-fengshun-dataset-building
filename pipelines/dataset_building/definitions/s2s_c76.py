"""S2S c76 daily product, with tcwv and ttr."""

from __future__ import annotations

from .standard_dataset import (
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_1P5,
    STANDARD_LAND_SEA_MASK,
)

from ._standard_products import PRESSURE_CHANNELS, CONST_SMALL, WEIGHT_NPY, define_channels

CHANNELS: tuple[str, ...] = PRESSURE_CHANNELS + (
    "t2m",
    "d2m",
    "sst",
    "ttr",
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "msl",
    "tcwv",
    "tp",
)
MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant("1p5.h24", GRID_1P5, 24, (CONST_SMALL, WEIGHT_NPY)),
)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="s2s.c76",
    version="1.0",
    channels=define_channels(CHANNELS, radiation_seconds=86400),
    variants=VARIANTS,
    mask=MASK,
)
