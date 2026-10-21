"""Discover and decode climatology fields, independently of evaluation contracts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

from datakit.io.readers.errors import SourceDecodeError, SourceDiscoveryError
from datakit.io.readers.hdf5 import hdf5_guard

DEFAULT_GRIB_FILTERS: dict[str, dict[str, Any]] = {
    "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
    "gh": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
    "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
    "msl": {"shortName": "msl", "typeOfLevel": "meanSea"},
    "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
    "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
}


class ClimatologyFiles:
    """Resolve valid times to files by month, day and hour, independent of year."""

    def __init__(
        self,
        root_dir: Path,
        template: str = "ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hh}.grib2",
    ):
        self.root_dir = Path(root_dir)
        self.template = template

    def file_name(self, moment: datetime) -> str:
        return self.template.format(
            mmdd=f"{moment.month:02d}{moment.day:02d}", hh=f"{moment.hour:02d}"
        )

    def discover_files(self, times: Sequence[datetime]) -> dict[datetime, Path]:
        if not self.root_dir.exists():
            raise SourceDiscoveryError(f"气候态根目录不存在: {self.root_dir}")
        if not times:
            raise SourceDiscoveryError("气候态请求需要 init_times（作为有效时刻）")
        mapping: dict[datetime, Path] = {}
        for moment in times:
            stem = (self.root_dir / self.file_name(moment)).with_suffix("")
            for suffix in (".grib2", ".grib", ".nc"):
                candidate = stem.with_suffix(suffix)
                if candidate.exists():
                    mapping[moment] = candidate
                    break
            else:
                key = f"{moment.month:02d}{moment.day:02d}"
                hour = f"{moment.hour:02d}"
                for pattern in (
                    f"*{key}{hour}*.grib2",
                    f"*{key}{hour}*.nc",
                    f"*{key}*.grib2",
                    f"*{key}*.nc",
                ):
                    matches = sorted(self.root_dir.glob(pattern))
                    if matches:
                        mapping[moment] = matches[0]
                        break
        if not mapping:
            raise SourceDiscoveryError(
                f"没有找到任何气候态文件（请求 {len(times)} 个时刻）"
            )
        return mapping


class ClimatologyDatasetReader:
    def __init__(
        self,
        engine: str = "cfgrib",
        grib_filters: dict[str, dict[str, Any]] | None = None,
    ):
        self.engine = engine
        self.grib_filters = grib_filters or DEFAULT_GRIB_FILTERS

    def read_files(
        self, mapping: dict[datetime, Path], variables: Sequence[str]
    ) -> tuple[xr.Dataset, list[str]]:
        frames = []
        input_files: list[str] = []
        for moment in sorted(mapping):
            try:
                dataset = self._read_one(mapping[moment], variables)
            except Exception as exc:
                raise SourceDecodeError(
                    f"Failed to read climatology file {mapping[moment]}: {exc}",
                    path=str(mapping[moment]),
                    cause=exc,
                ) from exc
            if dataset is None:
                continue
            frames.append(dataset.expand_dims(valid_time=[moment]))
            path = str(mapping[moment])
            if path not in input_files:
                input_files.append(path)
        if not frames:
            raise SourceDecodeError(f"气候态里没有请求的变量 {list(variables)}")
        combined = (
            frames[0]
            if len(frames) == 1
            else xr.concat(
                frames, dim="valid_time", coords="different", compat="equals"
            )
        )
        return combined, input_files

    def _read_one(self, path: Path, variables: Sequence[str]) -> xr.Dataset | None:
        datasets = []
        for variable in variables:
            name = str(variable)
            if self.engine == "cfgrib":
                filters = self.grib_filters.get(name)
                if filters is None:
                    continue
                with xr.open_dataset(
                    path, engine="cfgrib", backend_kwargs={"filter_by_keys": filters}
                ) as source:
                    dataset = source.load()
            else:
                with hdf5_guard(self.engine), xr.open_dataset(path) as source:
                    dataset = source.load()
                if name not in dataset.data_vars:
                    continue
                dataset = dataset[[name]]
            if name not in dataset.data_vars and len(dataset.data_vars) == 1:
                dataset = dataset.rename({next(iter(dataset.data_vars)): name})
            if name in dataset.data_vars:
                datasets.append(dataset[[name]])
        if not datasets:
            return None
        return datasets[0] if len(datasets) == 1 else xr.merge(datasets)
