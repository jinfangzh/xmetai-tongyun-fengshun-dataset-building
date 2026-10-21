"""Small anomaly batches shared by ACC and activity tests; no scoring logic."""

from __future__ import annotations

import xarray as xr

from datakit.evaluation.contracts import EvaluationBatch

ArrayLike = list[list[float]] | xr.DataArray


def make_grid_field(values: ArrayLike) -> xr.DataArray:
    if isinstance(values, xr.DataArray):
        return values
    return xr.DataArray(values, dims=("lat", "lon"))


def make_grid_batch(
    forecast: ArrayLike,
    observation: ArrayLike,
    climatology: ArrayLike | None,
    *,
    valid_mask: list[list[bool]] | xr.DataArray | None = None,
    weights: ArrayLike | None = None,
) -> EvaluationBatch:
    forecast_da = make_grid_field(forecast)
    observation_da = make_grid_field(observation)
    reference_da = make_grid_field(climatology) if climatology is not None else None
    if valid_mask is None:
        valid_mask_da = xr.ones_like(observation_da, dtype=bool)
    elif isinstance(valid_mask, xr.DataArray):
        valid_mask_da = valid_mask
    else:
        valid_mask_da = xr.DataArray(valid_mask, dims=observation_da.dims)
    weights_da = make_grid_field(weights) if weights is not None else None

    return EvaluationBatch(
        forecast=forecast_da,
        observation=observation_da,
        reference=reference_da,
        sample_keys=[{"case": "unit-test"}],
        valid_mask=valid_mask_da,
        weights=weights_da,
        alignment={"time": "same_valid_time", "space": "same_grid"},
        protocol_id="grid_valid_time",
    )
