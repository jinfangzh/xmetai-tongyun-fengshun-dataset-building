"""Run-bound dataset validation and source-to-target conversion specifications."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from datakit.schema.datasets.specifications import DatasetSchema, EvidenceReference

__all__ = ["DatasetValidationSpec", "ConversionChannelRule", "ConversionValidationSpec"]


@dataclass(frozen=True)
class DatasetValidationSpec(DatasetSchema):
    """One run's evidence references; never serialized back into its schema."""

    production_evidence: EvidenceReference | None = None
    snapshot_id: str | None = None
    conversion_evidence: EvidenceReference | None = None
    execution_config_path: Path | None = None
    execution_config_sha256: str | None = None

    @classmethod
    def from_schema(cls, schema: DatasetSchema) -> DatasetValidationSpec:
        return cls(**{item.name: getattr(schema, item.name) for item in fields(DatasetSchema)})

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if self.snapshot_id is not None and (
            not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip()
        ):
            raise ValueError("Snapshot identity must be a nonempty string")
        if (self.execution_config_path is None) != (self.execution_config_sha256 is None):
            raise ValueError("Execution configuration requires both source path and SHA256")
        if self.execution_config_path is not None and self.execution_config_sha256 is not None:
            EvidenceReference(self.execution_config_path, self.execution_config_sha256)


@dataclass(frozen=True)
class ConversionChannelRule:
    """Approved source-to-target linear mapping, applied to values once."""

    target_channel: str
    source_channel: str
    atol: float
    rtol: float
    value_factor: float = 1.0
    value_offset: float = 0.0
    mean_factor: float = 1.0
    mean_offset: float = 0.0
    std_factor: float = 1.0
    expected_mean: float | None = None
    expected_std: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.atol,
            self.rtol,
            self.value_factor,
            self.value_offset,
            self.mean_factor,
            self.mean_offset,
            self.std_factor,
        )
        if not np.isfinite(values).all() or self.atol < 0 or self.rtol < 0 or self.std_factor <= 0:
            raise ValueError("Conversion factors/tolerances must be finite and valid")


@dataclass(frozen=True)
class ConversionValidationSpec:
    source_identity: str
    source_snapshot_id: str
    rules_version: str
    source_first_time: str
    source_last_time: str
    source_cadence_hours: int
    source_channels: tuple[str, ...]
    rules: tuple[ConversionChannelRule, ...]
    source_variable: str = "data"
    source_channel_coordinate: str = "channel"
    source_dimensions: tuple[str, ...] = ("time", "channel", "lat", "lon")
    source_mean_variable: str = "mean"
    source_std_variable: str = "std"
    latitude_flip: bool = False
    source_statistics: EvidenceReference | None = None

    def expected_times(self) -> np.ndarray:
        start, end = (
            np.datetime64(self.source_first_time, "ns"),
            np.datetime64(self.source_last_time, "ns"),
        )
        if self.source_cadence_hours <= 0:
            raise ValueError("Invalid source cadence")
        step = np.timedelta64(self.source_cadence_hours, "h").astype("timedelta64[ns]")
        if np.isnat(start) or np.isnat(end) or end < start or (end - start) % step:
            raise ValueError("Invalid independently frozen source coverage")
        return np.arange(start, end + step, step)
