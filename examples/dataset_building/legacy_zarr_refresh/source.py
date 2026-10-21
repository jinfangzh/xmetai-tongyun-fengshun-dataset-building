"""Read-only adapter for legacy meteorological Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import xarray as xr
import zarr

from .channels import canonicalize_channels
from .config import RefreshProfile
CF_TIME = re.compile(
    r"^(?P<unit>hours?|days?|seconds?)\s+since\s+(?P<origin>.+)$",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_cf_time(values: np.ndarray, attributes: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    units = str(attributes.get("units", ""))
    match = CF_TIME.match(units.strip())
    if not match:
        raise ValueError(f"unsupported numeric time units: {units!r}")
    origin_text = match.group("origin").strip().replace(" UTC", "")
    origin = np.datetime64(origin_text).astype("datetime64[ns]")
    unit = match.group("unit").lower()
    code = "h" if unit.startswith("hour") else "D" if unit.startswith("day") else "s"
    numeric = np.asarray(values, dtype="i8")
    return origin + numeric.astype(f"timedelta64[{code}]").astype("timedelta64[ns]")


def metadata_file(path: Path) -> dict[str, Any]:
    v3 = path / "zarr.json"
    if v3.is_file():
        return json.loads(v3.read_text(encoding="utf-8"))
    v2 = path / ".zattrs"
    if v2.is_file():
        return {"attributes": json.loads(v2.read_text(encoding="utf-8")), "zarr_format": 2}
    raise FileNotFoundError(f"no Zarr root metadata found in {path}")


def correct_statistics(
    channels: tuple[str, ...],
    mean_source: np.ndarray,
    std_source: np.ndarray,
    divide: dict[str, float],
    multiply: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    mean = mean_source.astype("f4", copy=True)
    std = std_source.astype("f4", copy=True)
    for channel, divisor in divide.items():
        if channel not in channels:
            raise ValueError(f"statistics correction channel missing: {channel}")
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError(f"invalid statistics divisor for {channel}: {divisor}")
        if divisor == 1:
            continue
        index = channels.index(channel)
        mean[index] /= np.float32(divisor)
        std[index] /= np.float32(divisor)
    for pattern, factor in multiply.items():
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"invalid statistics multiplier for {pattern}: {factor}")
        indices = [index for index, name in enumerate(channels) if fnmatchcase(name, pattern)]
        if not indices:
            raise ValueError(f"statistics multiplier pattern matched no channels: {pattern}")
        if factor == 1:
            continue
        mean[indices] *= np.float32(factor)
        std[indices] *= np.float32(factor)
    return mean, std


@dataclass
class LegacyZarrSource:
    profile: RefreshProfile

    def __post_init__(self) -> None:
        self.path = self.profile.input
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        if self.profile.already_refreshed not in {"error", "force"}:
            raise ValueError("already_refreshed must be either 'error' or 'force'")
        previous_manifest = self.path / "refresh_manifest.json"
        if previous_manifest.is_file() and self.profile.already_refreshed != "force":
            raise ValueError(
                f"source already contains a refresh manifest: {previous_manifest}; "
                "set already_refreshed='force' only after verifying the source"
            )
        self.data = zarr.open_array(str(self.path / "data"), mode="r")
        coordinate = self.profile.channel_coordinate
        if coordinate == "auto":
            coordinate = "channel" if (self.path / "channel").is_dir() else "level"
        if coordinate not in {"channel", "level"}:
            raise ValueError(f"invalid channel coordinate {coordinate!r}")
        self.channel_coordinate = coordinate
        self.channel_array = zarr.open_array(str(self.path / coordinate), mode="r")
        self.lat_array = zarr.open_array(str(self.path / "lat"), mode="r")
        self.lon_array = zarr.open_array(str(self.path / "lon"), mode="r")
        self.time_array = zarr.open_array(str(self.path / "time"), mode="r")

        self.source_channels = tuple(_text(value) for value in self.channel_array[:].tolist())
        self.channels, self.channel_indices = canonicalize_channels(
            self.source_channels, self.profile.channel_rename
        )
        self.lat_source = np.asarray(self.lat_array[:], dtype="f8")
        self.lon = np.asarray(self.lon_array[:], dtype="f8")
        self.times = decode_cf_time(
            np.asarray(self.time_array[:]), dict(self.time_array.attrs)
        )
        mean_source = self._read_statistic("mean")
        std_source = self._read_statistic("std")
        order = np.asarray(self.channel_indices, dtype="i8")
        self.mean_source = mean_source[order]
        self.std_source = std_source[order]
        self.mean, self.std = self._correct_statistics()
        self.flip_latitude = bool(self.lat_source[0] < self.lat_source[-1])
        self.lat = self.lat_source[::-1].copy() if self.flip_latitude else self.lat_source.copy()
        self.mask = self._read_mask()
        if self.flip_latitude:
            self.mask = self.mask[::-1].copy()
        self._validate()

    def _read_statistic(self, statistic: str) -> np.ndarray:
        selected = self.profile.statistics
        candidates = []
        if selected == "auto":
            candidates = ["npy", "nc", "zarr", "root"]
        else:
            candidates = [selected]
        for backend in candidates:
            if backend == "npy" and (self.path / f"{statistic}.npy").is_file():
                return np.asarray(
                    np.load(self.path / f"{statistic}.npy"), dtype="f4"
                ).reshape(-1)
            if backend == "nc" and (self.path / f"{statistic}.nc").is_file():
                dataset = xr.open_dataset(self.path / f"{statistic}.nc")
                try:
                    variable = statistic if statistic in dataset.data_vars else next(
                        iter(dataset.data_vars)
                    )
                    values = np.asarray(dataset[variable].values, dtype="f4").reshape(-1)
                    if "channel" in dataset.coords:
                        labels = [_text(value) for value in dataset.channel.values.tolist()]
                        if len(labels) == len(values) and set(labels) == set(self.source_channels):
                            mapping = dict(zip(labels, values, strict=True))
                            values = np.asarray([mapping[name] for name in self.source_channels], dtype="f4")
                    return values
                finally:
                    dataset.close()
            if backend == "zarr" and (self.path / statistic).is_dir():
                array = zarr.open_array(str(self.path / statistic), mode="r")
                return np.asarray(array[:], dtype="f4").reshape(-1)
            if backend == "root":
                attributes = metadata_file(self.path).get("attributes", {})
                if statistic in attributes:
                    values = np.asarray(attributes[statistic], dtype="f4").reshape(-1)
                    channel_count = len(self.source_channels)
                    if values.size % channel_count != 0:
                        raise ValueError(
                            f"{statistic} root attribute length {values.size} is not "
                            f"a multiple of {channel_count} channels"
                        )
                    blocks = values.reshape(-1, channel_count)
                    if blocks.shape[0] > 1 and not np.allclose(
                        blocks, blocks[:1], rtol=0.0, atol=0.0
                    ):
                        raise ValueError(
                            f"{statistic} root attribute contains inconsistent repeated blocks"
                        )
                    return blocks[0]
        raise FileNotFoundError(f"cannot locate {statistic} for {self.path}")

    def _correct_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        return correct_statistics(
            self.channels,
            self.mean_source,
            self.std_source,
            self.profile.statistics_divide,
            self.profile.statistics_multiply,
        )
    def _read_mask(self) -> np.ndarray:
        spec = self.profile.mask
        kind = str(spec.get("kind", ""))
        relative = Path(str(spec.get("path", "")))
        path = self.path / relative
        if kind == "npy":
            values = np.load(path)
        elif kind == "netcdf":
            dataset = xr.open_dataset(path)
            try:
                variable = str(spec["variable"])
                values = np.asarray(dataset[variable].values)
            finally:
                dataset.close()
        elif kind == "const_channel":
            dataset = xr.open_dataset(path)
            try:
                variable = str(spec["variable"])
                channel = str(spec.get("channel", "lsm"))
                field = dataset[variable]
                if "channel" not in field.dims:
                    raise ValueError(f"{path}: {variable} has no channel dimension")
                values = np.asarray(field.sel(channel=channel).values)
            finally:
                dataset.close()
        else:
            raise ValueError(f"unsupported mask kind {kind!r} for {self.profile.key}")
        values = np.asarray(values).squeeze()
        if values.shape != tuple(self.data.shape[-2:]):
            raise ValueError(
                f"mask shape {values.shape} differs from data grid {self.data.shape[-2:]}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"mask contains non-finite values: {path}")
        if values.dtype == np.bool_ or np.isin(values, (0, 1)).all():
            self.mask_binarized = False
            return values.astype("u1")
        threshold = float(spec.get("threshold", 0.5))
        if not np.isfinite(threshold):
            raise ValueError(f"mask threshold must be finite, got {threshold}")
        self.mask_binarized = True
        return (values > threshold).astype("u1")

    def _validate(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(f"data must be 4-D, got {self.data.shape}")
        expected = (len(self.times), len(self.channels), len(self.lat), len(self.lon))
        if tuple(self.data.shape) != expected:
            raise ValueError(f"data shape {self.data.shape} != coordinates {expected}")
        if self.profile.expected_channels is not None and len(self.channels) != self.profile.expected_channels:
            raise ValueError("unexpected channel count")
        if self.profile.expected_grid is not None and tuple(self.data.shape[-2:]) != self.profile.expected_grid:
            raise ValueError("unexpected grid shape")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channel labels are not unique")
        if self.mean.shape != (len(self.channels),) or self.std.shape != (len(self.channels),):
            raise ValueError("mean/std are not aligned with channels")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("mean/std contain invalid values")
        if len(self.times) > 1:
            differences = np.diff(self.times)
            expected_step = np.timedelta64(self.profile.cadence_hours, "h")
            if not np.all(differences == expected_step):
                bad = np.flatnonzero(differences != expected_step)[:10].tolist()
                raise ValueError(f"time cadence mismatch at indices {bad}")
        if not np.all(np.diff(self.lon) > 0):
            raise ValueError("longitude must be strictly increasing")
        if not np.all(np.diff(self.lat) < 0):
            raise ValueError("target latitude must be strictly decreasing")
        if not np.isin(self.mask, (0, 1)).all():
            raise ValueError("mask is not binary")

    def read_data(self, index: int) -> np.ndarray:
        values = np.asarray(self.data[index], dtype="f2")
        identity = tuple(range(len(self.channel_indices)))
        if self.channel_indices != identity:
            values = values[np.asarray(self.channel_indices, dtype="i8")]
        if self.flip_latitude:
            values = values[:, ::-1, :]
        return np.ascontiguousarray(values)

    def fingerprint(self) -> str:
        payload = {
            "profile": self.profile.serializable(),
            "shape": tuple(int(v) for v in self.data.shape),
            "chunks": tuple(int(v) for v in self.data.chunks),
            "source_channels": self.source_channels,
            "channels": self.channels,
            "time_first": str(self.times[0]),
            "time_last": str(self.times[-1]),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def describe(self) -> dict[str, Any]:
        radiation = {}
        for name in ("ssr", "ssrd", "fdir", "ttr"):
            if name in self.channels:
                index = self.channels.index(name)
                radiation[name] = {
                    "source_mean": float(self.mean_source[index]),
                    "source_std": float(self.std_source[index]),
                    "target_mean": float(self.mean[index]),
                    "target_std": float(self.std[index]),
                }
        return {
            "key": self.profile.key,
            "input": str(self.path),
            "data_shape": [int(v) for v in self.data.shape],
            "data_chunks": [int(v) for v in self.data.chunks],
            "data_dtype": str(self.data.dtype),
            "channel_coordinate": self.channel_coordinate,
            "channel_count": len(self.channels),
            "source_channels": list(self.source_channels),
            "channels": list(self.channels),
            "channel_source_indices": list(self.channel_indices),
            "grid": [len(self.lat), len(self.lon)],
            "source_latitude_order": "ascending" if self.flip_latitude else "descending",
            "target_latitude_order": "descending",
            "time_count": len(self.times),
            "time_start": str(self.times[0]),
            "time_end": str(self.times[-1]),
            "cadence_hours": self.profile.cadence_hours,
            "radiation_seconds": self.profile.radiation_seconds,
            "radiation_statistics": radiation,
            "mask_land_fraction": float(self.mask.mean()),
            "fingerprint": self.fingerprint(),
        }
