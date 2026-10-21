"""Versioned expectations for standardized datasets, without file IO.

Products define semantics. Instance specifications define independently frozen
coverage and quality rules. Neither is inferred from the object being checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class GridDefinition:
    name: str
    latitude_count: int
    longitude_count: int
    resolution: float
    north: float = 90.0
    west: float = 0.0
    latitude_direction: str = "north-to-south"

    def __post_init__(self) -> None:
        if (
            not self.name
            or min(self.latitude_count, self.longitude_count) < 1
            or not np.isfinite((self.resolution, self.north, self.west)).all()
            or self.resolution <= 0
            or self.latitude_direction not in ("north-to-south", "south-to-north")
        ):
            raise ValueError("Grid dimensions and resolution must be positive and finite")
        if (
            self.north > 90
            or self.north - (self.latitude_count - 1) * self.resolution < -90
            or self.west < 0
            or self.west + (self.longitude_count - 1) * self.resolution >= 360
        ):
            raise ValueError("Grid must stay within -90..90 latitude and unique 0..360 longitude")

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        latitude = self.north - np.arange(self.latitude_count, dtype="f4") * self.resolution
        if self.latitude_direction == "south-to-north":
            latitude = latitude[::-1].copy()
        return (
            latitude,
            self.west + np.arange(self.longitude_count, dtype="f4") * self.resolution,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.latitude_count, self.longitude_count


GRID_P25 = GridDefinition("p25", 721, 1440, 0.25)
GRID_CRA_C71_P25 = GridDefinition("p25", 721, 1440, 0.25, latitude_direction="south-to-north")
GRID_1P5 = GridDefinition("1p5", 121, 240, 1.5)


@dataclass(frozen=True)
class MaskDefinition:
    """Embedded mask contract; spatial size and direction come from the grid."""

    schema_id: str = "xmetai.land-sea-mask"
    version: str = "1.0"
    array_name: str = "mask"
    channel_array_name: str = "mask_channel"
    dimensions: tuple[str, ...] = ("mask_channel", "lat", "lon")
    channel_dimensions: tuple[str, ...] = ("mask_channel",)
    dtype: str = "uint8"
    channel_dtype: str = "string"
    channels: tuple[str, ...] = ("land_mask", "sea_mask")
    land_channel: str = "land_mask"
    sea_channel: str = "sea_mask"
    allowed_values: tuple[int, ...] = (0, 1)
    valid_value: int = 1
    complement_sum: int = 1
    require_provenance: bool = False
    check_external_masks: bool = False

    def __post_init__(self) -> None:
        reserved = {"data", "time", "channel", "lat", "lon", "mean", "std"}
        names = (self.array_name, self.channel_array_name)
        if (
            not self.schema_id
            or not self.version
            or names[0] == names[1]
            or any(not name or "/" in name or "\\" in name or name in reserved for name in names)
            or len(self.channel_dimensions) != 1
            or len(set(self.dimensions)) != 3
            or self.dimensions != (*self.channel_dimensions, "lat", "lon")
            or self.channel_dtype != "string"
        ):
            raise ValueError("Invalid embedded mask names or channel-first spatial layout")
        if (
            len(self.channels) != 2
            or len(set(self.channels)) != 2
            or self.land_channel == self.sea_channel
            or set(self.channels) != {self.land_channel, self.sea_channel}
            or not all(isinstance(name, str) and name for name in self.channels)
        ):
            raise ValueError("Mask channels must uniquely declare the land and sea roles")
        if (
            np.dtype(self.dtype).kind not in "biu"
            or len(self.allowed_values) != 2
            or len(set(self.allowed_values)) != 2
            or any(type(value) is not int for value in self.allowed_values)
            or self.valid_value not in self.allowed_values
            or type(self.valid_value) is not int
            or type(self.complement_sum) is not int
            or sum(self.allowed_values) != self.complement_sum
            or type(self.require_provenance) is not bool
            or type(self.check_external_masks) is not bool
        ):
            raise ValueError("Invalid binary mask values, complement or acceptance scope")
        bounds = (
            (0, 1)
            if np.dtype(self.dtype).kind == "b"
            else (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)
        )
        if not all(bounds[0] <= value <= bounds[1] for value in self.allowed_values):
            raise ValueError("Mask values must be representable in the declared dtype")

    def shape(self, grid: GridDefinition) -> tuple[int, int, int]:
        return len(self.channels), *grid.shape

    def layouts(
        self, grid: GridDefinition
    ) -> dict[str, tuple[tuple[str, ...], tuple[int, ...], str]]:
        return {
            self.array_name: (self.dimensions, self.shape(grid), self.dtype),
            self.channel_array_name: (
                self.channel_dimensions,
                (len(self.channels),),
                self.channel_dtype,
            ),
        }

    def region_index(self, region: str) -> int:
        if region not in ("land", "sea"):
            raise ValueError(f"Undefined embedded mask region: {region}")
        return self.channels.index(self.land_channel if region == "land" else self.sea_channel)


STANDARD_LAND_SEA_MASK = MaskDefinition()


@dataclass(frozen=True)
class ChannelDefinition:
    name: str
    variable: str
    units: str
    source_units: str
    level_type: str
    level: float | int | None = None
    level_units: str | None = None
    radiation_seconds: int | None = None
    variable_type: str = "instantaneous"
    source_name: str | tuple[str, ...] | None = None
    unconfirmed_fields: tuple[str, ...] = ()

    def semantic_fields(self) -> dict[str, Any]:
        zscore = {"operation": "zscore", "mean": "/mean", "std": "/std"}
        if self.variable == "tp":
            steps = [
                {
                    "operation": "multiply",
                    "factor": 1000.0,
                    "input_units": "m",
                    "output_units": "mm",
                },
                {"operation": "clip_min", "minimum": 0.0},
                {"operation": "log1p"},
            ]
        elif self.variable in ("q", "q2m"):
            steps = [
                {
                    "operation": "multiply",
                    "factor": 1000.0,
                    "input_units": "kg kg-1",
                    "output_units": "g kg-1",
                },
                zscore,
            ]
        elif self.variable in ("ssr", "ssrd", "fdir", "ttr"):
            if not self.radiation_seconds:
                raise ValueError(f"Missing accumulation window: {self.name}")
            steps = [
                {
                    "operation": "divide",
                    "divisor": float(self.radiation_seconds),
                    "accumulation_window_seconds": self.radiation_seconds,
                    "input_units": "J m-2",
                    "output_units": "W m-2",
                },
                zscore,
            ]
        elif self.variable in ("ws10m", "ws100m"):
            height = 10 if self.variable == "ws10m" else 100
            steps = [{"operation": "hypot", "inputs": [f"u{height}m", f"v{height}m"]}, zscore]
        else:
            steps = [zscore]
        result = {
            "units": self.units,
            "source_units": self.source_units,
            "level_type": self.level_type,
            "variable_type": self.variable_type,
            "preprocessing": steps,
        }
        source_name = self.source_name or self.variable
        result["source_name"] = list(source_name) if isinstance(source_name, tuple) else source_name
        if self.level is not None:
            result.update(level=self.level, level_units=self.level_units)
        return result


def build_standard_channel_info(
    product: DatasetProduct, *, mean: np.ndarray, std: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Generate declarations from a selected product, never parse unknown labels."""
    count = len(product.channels)
    if (
        mean.shape != (count,)
        or std.shape != (count,)
        or mean.dtype != np.dtype("f4")
        or std.dtype != np.dtype("f4")
    ):
        raise ValueError("Statistics must be raw float32 vectors matching product channels")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Statistics must be finite with positive std")
    result = {}
    for index, channel in enumerate(product.channels):
        if channel.variable == "tp" and (mean[index] != 0 or std[index] != 1):
            raise ValueError("Unnormalized log1p precipitation requires effective mean=0/std=1")
        fields = channel.semantic_fields()
        fields.update(
            long_name=channel.name,
            scale_factor=float(std[index]),
            add_offset=float(mean[index]),
            scale_offset_formula="preprocessed_value = normalized_value * scale_factor + add_offset",
        )
        result[channel.name] = fields
    return result


@dataclass(frozen=True)
class AuxiliaryDefinition:
    filename: str
    format: str
    variable: str | None
    dimensions: tuple[str, ...]
    dtype: str
    role: str
    channels: tuple[str, ...] = ()
    required: bool = True
    semantics_confirmed: bool = True
    equals_mask: str | None = None
    check_variable_name: bool = False
    weight_nonnegative_atol: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.weight_nonnegative_atol, Real)
            or isinstance(self.weight_nonnegative_atol, (bool, np.bool_))
            or not np.isfinite(self.weight_nonnegative_atol)
            or self.weight_nonnegative_atol < 0
        ):
            raise ValueError("Weight nonnegative tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class DatasetVariant:
    name: str
    grid: GridDefinition
    cadence_hours: int
    auxiliaries: tuple[AuxiliaryDefinition, ...] = ()


@dataclass(frozen=True)
class DatasetProduct:
    product_id: str
    version: str
    channels: tuple[ChannelDefinition, ...]
    variants: tuple[DatasetVariant, ...]
    semantic_review: tuple[str, ...] = ()
    mask: MaskDefinition = STANDARD_LAND_SEA_MASK

    def __post_init__(self) -> None:
        if not isinstance(self.mask, MaskDefinition):
            raise ValueError("A product must declare an embedded mask schema")
        names = self.channel_names
        if not names or len(set(names)) != len(names):
            raise ValueError("Product channel names must be nonempty and unique")
        if not self.variants or len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("Product variants must be nonempty and unique")
        for channel in self.channels:
            if len(set(channel.unconfirmed_fields)) != len(channel.unconfirmed_fields) or set(
                channel.unconfirmed_fields
            ) - {"units", "source_units"}:
                raise ValueError("Unconfirmed semantic fields must be explicit unit fields")
            if channel.units == "unknown" or channel.source_units == "unknown":
                raise ValueError(f"Undefined channel semantics: {channel.name}")
            channel.semantic_fields()

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.channels)

    def variant(self, name: str) -> DatasetVariant:
        for item in self.variants:
            if item.name == name:
                return item
        raise ValueError(f"Unknown variant {name!r} for {self.product_id}")


@dataclass(frozen=True)
class QualityRule:
    channel: str
    valid_region: str
    max_nan_fraction: float
    allow_constant: bool
    evidence: str
    bounds: tuple[float, float] | None = None
    domain: str = "preprocessed"
    max_block_nan_fraction: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.channel, str)
            or not self.channel
            or not isinstance(self.valid_region, str)
            or not isinstance(self.evidence, str)
            or type(self.allow_constant) is not bool
        ):
            raise ValueError(
                "Quality names/evidence must be strings and allow_constant must be boolean"
            )
        if not isinstance(self.max_nan_fraction, Real) or isinstance(
            self.max_nan_fraction, (bool, np.bool_)
        ):
            raise ValueError("NaN thresholds must be numbers, not booleans or strings")
        if self.max_block_nan_fraction is not None and (
            not isinstance(self.max_block_nan_fraction, Real)
            or isinstance(self.max_block_nan_fraction, (bool, np.bool_))
        ):
            raise ValueError("Per-block NaN thresholds must be numeric")
        if (
            self.valid_region not in ("all", "land", "sea")
            and not self.valid_region.startswith("custom:")
        ) or self.domain not in (
            "stored",
            "preprocessed",
            "physical",
        ):
            raise ValueError("Invalid quality region or domain")
        if not 0 <= self.max_nan_fraction <= 1 or not self.evidence:
            raise ValueError("A quality rule requires a valid threshold and independent evidence")
        if self.max_block_nan_fraction is not None and not 0 <= self.max_block_nan_fraction <= 1:
            raise ValueError("Per-block NaN tolerance must be in [0,1]")
        if self.bounds is not None and (
            len(self.bounds) != 2
            or any(
                not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                for value in self.bounds
            )
            or not np.isfinite(self.bounds).all()
            or self.bounds[0] > self.bounds[1]
        ):
            raise ValueError("Quality bounds must be finite and ordered")


@dataclass(frozen=True)
class EvidenceReference:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256.lower())
        ):
            raise ValueError("Evidence requires an independently frozen SHA256")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True)
class DatasetValidationSpec:
    dataset_id: str
    product: DatasetProduct
    variant_name: str
    first_time: str
    last_time: str
    data_revision: str
    profile_version: str = "1.0"
    contract_version: str = "1.0"
    schema_version: str = "2.0"
    content_version: str = "v3"
    zarr_format: int = 3
    quality: tuple[QualityRule, ...] = ()
    mask_provenance: str | None = None
    mask_derivation: str | None = None
    auxiliary_provenance: Mapping[str, str] = field(default_factory=dict)
    auxiliary_tolerances: Mapping[str, float] = field(default_factory=dict)
    confirmed_optional_roles: Mapping[str, str] = field(default_factory=dict)
    production_evidence: EvidenceReference | None = None
    snapshot_id: str | None = None
    allowed_omitted: tuple[tuple[int, ...], ...] = ()
    consumer_required: bool = False
    conversion_required: bool = False
    semantic_confirmation: str | None = None
    directory_name: str | None = None
    identity_mapping_evidence: str | None = None
    consumer_evidence: EvidenceReference | None = None
    conversion_evidence: EvidenceReference | None = None
    additional_metadata_files: tuple[str, ...] = ("auxiliary_registry.json",)
    valid_regions: Mapping[str, EvidenceReference] = field(default_factory=dict)
    valid_region_provenance: Mapping[str, str] = field(default_factory=dict)
    validation_scope: str = "strict-standard"

    @property
    def variant(self) -> DatasetVariant:
        return self.product.variant(self.variant_name)

    def expected_times(self) -> np.ndarray:
        start, end = np.datetime64(self.first_time, "ns"), np.datetime64(self.last_time, "ns")
        step = np.timedelta64(self.variant.cadence_hours, "h").astype("timedelta64[ns]")
        if np.isnat(start) or np.isnat(end) or end < start or (end - start) % step:
            raise ValueError("Coverage must have valid, cadence-aligned endpoints")
        if (
            start.astype("datetime64[h]").astype("datetime64[ns]") != start
            or int(start.astype("datetime64[h]").astype("int64")) % self.variant.cadence_hours
        ):
            raise ValueError("Expected UTC phase is 00/06/12/18 or daily 00")
        return np.arange(start, end + step, step)

    @property
    def expected_shape(self) -> tuple[int, int, int, int]:
        return (len(self.expected_times()), len(self.product.channels), *self.variant.grid.shape)

    @property
    def expected_chunks(self) -> tuple[int, int, int, int]:
        return (1, *self.expected_shape[1:])

    def validate_configuration(self) -> None:
        self.expected_times()
        if self.validation_scope not in ("strict-standard", "confirmed-format"):
            raise ValueError("Unsupported validation scope")
        if self.validation_scope == "confirmed-format" and self.quality:
            raise ValueError(
                "Confirmed-format profiles do not include provisional quality policies"
            )
        if (
            not self.dataset_id
            or not self.data_revision
            or self.zarr_format not in (2, 3)
            or self.contract_version != "1.0"
        ):
            raise ValueError("Incomplete or unsupported instance identity/contract")
        rules = [r.channel for r in self.quality]
        used_regions = {
            rule.valid_region.removeprefix("custom:")
            for rule in self.quality
            if rule.valid_region.startswith("custom:")
        }
        if used_regions != set(self.valid_regions) or any(
            not name or not self.valid_region_provenance.get(name) for name in used_regions
        ):
            raise ValueError(
                "Custom valid regions require exact frozen references and grid/source provenance"
            )
        auxiliaries = {definition.filename for definition in self.variant.auxiliaries}
        if any(
            Path(name).name != name or "/" in name or "\\" in name
            for name in self.additional_metadata_files
        ):
            raise ValueError("Additional metadata files must be explicit direct-child names")
        if any(
            name not in auxiliaries
            or not np.isfinite(tolerance)
            or tolerance < 0
            or not self.auxiliary_provenance.get(name)
            for name, tolerance in self.auxiliary_tolerances.items()
        ):
            raise ValueError(
                "Auxiliary tolerances require declared files, finite nonnegative values and independent provenance"
            )
        if len(set(rules)) != len(rules) or set(rules) - set(self.product.channel_names):
            raise ValueError("Quality rules duplicate or reference unknown channels")
        if len(set(self.allowed_omitted)) != len(self.allowed_omitted):
            raise ValueError("Duplicate approved omissions")
        if self.mask_derivation not in (None, "lsm>0.5"):
            raise ValueError("Unsupported explicitly approved mask derivation")
        for index in self.allowed_omitted:
            if (
                len(index) != 4
                or not 0 <= index[0] < self.expected_shape[0]
                or index[1:] != (0, 0, 0)
            ):
                raise ValueError("Approved omissions must use expected logical block indices")


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
        start, end = np.datetime64(self.source_first_time, "ns"), np.datetime64(
            self.source_last_time, "ns"
        )
        if self.source_cadence_hours <= 0:
            raise ValueError("Invalid source cadence")
        step = np.timedelta64(self.source_cadence_hours, "h").astype("timedelta64[ns]")
        if np.isnat(start) or np.isnat(end) or end < start or (end - start) % step:
            raise ValueError("Invalid independently frozen source coverage")
        return np.arange(start, end + step, step)
