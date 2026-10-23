"""Build evaluation bundles from prepared datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from datakit.evaluation.contracts import (
    DataBundle,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from datakit.schema.mapping import DatasetMapping
from datakit.schema.semantics import DataKind, TemporalKind


def build_bundle(
    request: DataRequest,
    dataset,
    input_files: Sequence[str],
    reader_id: str,
    reader_version: str,
    *,
    kind: DataKind,
    mapping: DatasetMapping | None = None,
    grid_type: str = "regular_latlon",
    temporal_kind=None,
    prefer_dataset_units: bool = True,
    timezone: str = "UTC",
    standard_vars=None,
    source_variables=None,
) -> DataBundle:
    """Wrap prepared data with schema metadata and the caller's evaluation role."""
    units, temporal, names = {}, {}, {}
    for name, variable in dataset.data_vars.items():
        name = str(name)
        canonical = name
        if mapping is not None and mapping.spec_for(name) is None:
            canonical = mapping.standard_name_for(name) or name
        spec = mapping.spec_for(canonical) if mapping is not None else None
        declared = spec.unit if spec is not None else "unknown"
        units[name] = str(
            (variable.attrs.get("units") or declared)
            if prefer_dataset_units
            else declared
        )
        temporal[name] = (
            spec.temporal_kind if spec is not None else TemporalKind.INSTANTANEOUS
        )
        names[name] = canonical
    if isinstance(temporal_kind, Mapping):
        temporal = {
            name: temporal_kind.get(name, TemporalKind.INSTANTANEOUS) for name in names
        }
    elif temporal_kind is not None:
        temporal = temporal_kind
    elif mapping is None:
        temporal = TemporalKind.INSTANTANEOUS
    member_dim = mapping.member_dim if mapping is not None else None
    return DataBundle(
        payload=dataset,
        kind=kind,
        source_id=request.source_id,
        standard_vars=dict(standard_vars) if standard_vars is not None else names,
        semantic=SemanticMetadata(
            units=units,
            temporal_kind=temporal,
            grid_type=grid_type,
            member_count=int(dataset.sizes[member_dim])
            if member_dim in dataset.sizes
            else None,
            timezone=timezone,
        ),
        provenance=Provenance(
            input_files=list(dict.fromkeys(input_files)),
            reader_id=reader_id,
            reader_version=reader_version,
            source_variables=dict(source_variables)
            if source_variables is not None
            else None,
        ),
    )
