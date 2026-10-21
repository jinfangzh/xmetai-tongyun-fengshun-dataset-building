"""Declarative rules for presenting decoded source data to evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from datakit.schema.semantics import DataKind, TemporalKind


@dataclass(frozen=True)
class AdaptationRules:
    """Metadata and names expected by evaluation, without reading or changing data."""

    kind: DataKind
    grid_type: str
    variable_names: Mapping[str, str] = field(default_factory=dict)
    coordinate_names: Mapping[str, str] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    prefer_dataset_units: bool = False
    temporal_kind: Mapping[str, TemporalKind] = field(default_factory=dict)
    default_temporal_kind: TemporalKind = TemporalKind.INSTANTANEOUS
    temporal_per_variable: bool = False
    member_dim: str | None = None
