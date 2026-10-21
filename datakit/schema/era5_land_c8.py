"""ERA5-Land four soil layers."""

from __future__ import annotations

from .standard_dataset import (
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_1P5,
    STANDARD_LAND_SEA_MASK,
)

from ._standard_products import WEIGHT_NPY, define_channels

# Freeze the native order of ERA5-Land/zarr/land.1950-2024.c8.
CHANNELS: tuple[str, ...] = tuple(
    f"{variable}{i}" for variable in ("stl", "swvl") for i in range(1, 5)
)
MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
VARIANTS: tuple[DatasetVariant, ...] = (DatasetVariant("1p5.h24", GRID_1P5, 24, (WEIGHT_NPY,)),)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="era5-land.c8",
    version="1.0",
    channels=define_channels(CHANNELS),
    variants=VARIANTS,
    mask=MASK,
)
