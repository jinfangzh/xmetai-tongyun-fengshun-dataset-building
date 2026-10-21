"""Dataset-building definitions, private to this workflow until shared contracts mature."""

from .standard_dataset import (
    STANDARD_LAND_SEA_MASK,
    AuxiliaryDefinition,
    ChannelDefinition,
    ConversionChannelRule,
    ConversionValidationSpec,
    DatasetProduct,
    DatasetValidationSpec,
    DatasetVariant,
    EvidenceReference,
    GridDefinition,
    MaskDefinition,
    QualityRule,
    build_standard_channel_info,
)
from .dataset_products import STANDARD_PRODUCTS, get_dataset_product

__all__ = [
    "STANDARD_LAND_SEA_MASK",
    "AuxiliaryDefinition",
    "ChannelDefinition",
    "ConversionChannelRule",
    "ConversionValidationSpec",
    "DatasetProduct",
    "DatasetValidationSpec",
    "DatasetVariant",
    "EvidenceReference",
    "GridDefinition",
    "MaskDefinition",
    "QualityRule",
    "build_standard_channel_info",
    "STANDARD_PRODUCTS",
    "get_dataset_product",
]
