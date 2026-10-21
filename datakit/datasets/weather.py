from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from datakit.io import read_netcdf, read_zarr


PathLike = str | Path
SampleFormat = Literal["numpy", "xarray"]

__all__ = [
	"NetCDFReader",
	"WeatherDataset",
	"WeatherReader",
	"WeatherSample",
	"ZarrReader",
]


def _select_data_array(dataset: xr.Dataset | xr.DataArray, variable: str) -> xr.DataArray:
	if isinstance(dataset, xr.DataArray):
		return dataset
	if variable in dataset:
		return dataset[variable]
	if len(dataset.data_vars) == 1:
		return dataset[next(iter(dataset.data_vars))]
	raise KeyError(
		f"Variable {variable!r} not found; available: {sorted(dataset.data_vars)}"
	)


def _normalize_fcst_frames(fcst_frames: int | Sequence[int]) -> np.ndarray:
	values = np.asarray(fcst_frames, dtype=int).reshape(-1)
	if values.size == 0:
		raise ValueError("fcst_frames cannot be empty")
	if (values < 1).any():
		raise ValueError("fcst_frames values must be >= 1")
	return values


class WeatherReader(ABC):
	"""Format-specific reader interface used by WeatherDataset."""

	def __init__(self, source: PathLike):
		self.source = source

	@abstractmethod
	def read(self, *, variable: str = "data", **kwargs: Any) -> xr.DataArray:
		"""Return a weather data array with a time dimension."""


class ZarrReader(WeatherReader):
	"""Read weather data from a Zarr store."""

	def __init__(self, source: PathLike, *, consolidated: bool | None = True):
		super().__init__(source)
		self.consolidated = consolidated

	def read(self, *, variable: str = "data", **kwargs: Any) -> xr.DataArray:
		dataset = read_zarr(self.source, consolidated=self.consolidated, **kwargs)
		return _select_data_array(dataset, variable)


class NetCDFReader(WeatherReader):
	"""Read weather data from a NetCDF file."""

	def read(self, *, variable: str = "data", **kwargs: Any) -> xr.DataArray:
		dataset = read_netcdf(self.source, **kwargs)
		return _select_data_array(dataset, variable)


@dataclass(frozen=True)
class WeatherSample:
	"""One sequence sample for training/inference preprocessing."""

	times: np.ndarray | xr.DataArray
	inputs: np.ndarray | xr.DataArray
	targets: np.ndarray | xr.DataArray | None


class WeatherDataset:
	"""Build sliding-window weather samples from a WeatherReader."""

	def __init__(
		self,
		reader: WeatherReader,
		*,
		variable: str = "data",
		hist_frames: int = 1,
		fcst_frames: int | Sequence[int] = 1,
		stride: int = 1,
		inference_only: bool = False,
		years: tuple[str, str] | None = None,
		sample_format: SampleFormat = "numpy",
		read_kwargs: dict[str, Any] | None = None,
	):
		if hist_frames < 1:
			raise ValueError("hist_frames must be >= 1")
		if stride < 1:
			raise ValueError("stride must be >= 1")
		if sample_format not in ("numpy", "xarray"):
			raise ValueError("sample_format must be 'numpy' or 'xarray'")

		self.reader = reader
		self.variable = variable
		self.hist_frames = int(hist_frames)
		self.fcst_frames = _normalize_fcst_frames(fcst_frames)
		self.inference_only = inference_only
		self.stride = int(stride)
		self.sample_format = sample_format

		data = self.reader.read(variable=variable, **(read_kwargs or {}))
		if years is not None:
			data = data.sel(time=slice(*years))
		if "time" not in data.dims:
			raise ValueError("Weather data must contain a 'time' dimension")
		self.data = data

		self.max_lead = 0 if self.inference_only else int(self.fcst_frames.max())
		self.total_frames = self.hist_frames + self.max_lead

		time_size = int(self.data.sizes["time"])
		if time_size < self.total_frames:
			raise ValueError(
				f"Not enough time steps: {time_size} < required {self.total_frames}"
			)
		self.starts = np.arange(0, time_size - self.total_frames + 1, self.stride, dtype=int)

	def __len__(self) -> int:
		return int(self.starts.size)

	def __getitem__(self, index: int) -> WeatherSample:
		start = int(self.starts[index])
		stop = start + self.total_frames
		window = self.data.isel(time=slice(start, stop))

		inputs = window.isel(time=slice(0, self.hist_frames))
		if self.inference_only:
			targets = None
			times = inputs["time"]
		else:
			target_positions = self.hist_frames - 1 + self.fcst_frames
			targets = window.isel(time=target_positions)
			times = window["time"]

		if self.sample_format == "xarray":
			return WeatherSample(times=times, inputs=inputs, targets=targets)

		np_times = np.asarray(times.values)
		np_inputs = np.asarray(inputs.values)
		np_targets = None if targets is None else np.asarray(targets.values)
		return WeatherSample(times=np_times, inputs=np_inputs, targets=np_targets)



