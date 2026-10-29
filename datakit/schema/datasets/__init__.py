"""Generic dataset schema models and strict loading of caller-supplied YAML."""

from .inputs import load_schema, resolve_config_path, verify_schema_unchanged
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
