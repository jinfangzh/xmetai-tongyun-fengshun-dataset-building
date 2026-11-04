"""Schema models, strict YAML loading, dimensions and variable metadata."""

from .mapping import DatasetMapping, VarSpec, load_mapping
from .dimensions import (
    DIMENSION_ALIASES,
    LEVELS_13,
)
from .variables import (
    ACCUMULATED_VARIABLES,
    SHORT_NAME_MAPPING,
    UNIT_SCALE,
    WATER_VARIABLES,
)

from .parser import load_schema, resolve_config_path, verify_schema_unchanged
from .specifications import (
    ArrayDefinition,
    AuxiliaryDefinition,
    ChannelDefinition,
    ChannelMetadataDefinition,
    DatasetProduct,
    DatasetSchema,
    DatasetVariant,
    EvidenceReference,
    GridDefinition,
    MaskDefinition,
    QualityRule,
    build_standard_channel_info,
)

__all__ = [
    "DatasetMapping",
    "VarSpec",
    "load_mapping",
    "ACCUMULATED_VARIABLES",
    "DIMENSION_ALIASES",
    "LEVELS_13",
    "SHORT_NAME_MAPPING",
    "UNIT_SCALE",
    "WATER_VARIABLES",
    "ArrayDefinition",
    "AuxiliaryDefinition",
    "ChannelDefinition",
    "ChannelMetadataDefinition",
    "DatasetProduct",
    "DatasetSchema",
    "DatasetVariant",
    "EvidenceReference",
    "GridDefinition",
    "MaskDefinition",
    "QualityRule",
    "build_standard_channel_info",
    "load_schema",
    "resolve_config_path",
    "verify_schema_unchanged",
]
