"""Check the evaluation transforms after moving them into datakit.transforms."""

from datetime import datetime

import numpy as np
import xarray as xr

from datakit.evaluation.contracts import DataBundle, Provenance, SemanticMetadata
from datakit.evaluation.ensemble import compute_ensemble_mean as mean_bundle
from datakit.evaluation.matching import _latitude_weights
from datakit.schema.semantics import DataKind, TemporalKind
from datakit.transforms import (
    GridToStationInterpolator,
    RegularLatLonRegridder,
    TimeWindowAccumulator,
    align_to_grid,
    resize,
    window_sum_at,
)


def test_grid_alignment_and_station_interpolation():
    grid = xr.Dataset(
        {"value": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [20.0, 10.0], "lon": [100.0, 110.0]},
    )
    aligned = align_to_grid(grid, [10.0, 20.0], [100.0, 110.0])
    assert aligned is not None
    np.testing.assert_array_equal(aligned.value.values, [[3.0, 4.0], [1.0, 2.0]])

    station = GridToStationInterpolator("bilinear").transform(
        grid.value, np.array([15.0]), np.array([105.0])
    )
    np.testing.assert_allclose(station.values, [2.5])


def test_temporal_window_preserves_missing_data_rules():
    observation = xr.Dataset(
        {"rain": (("time", "station"), [[1.0, np.nan], [2.0, 3.0]])},
        coords={"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "station": [0, 1]},
    )
    result = window_sum_at(observation, "rain", datetime(2026, 1, 1, 1), 2)
    np.testing.assert_allclose(result.values, [3.0, np.nan])
    permissive = window_sum_at(
        observation, "rain", datetime(2026, 1, 1, 1), 2, require_complete=False
    )
    np.testing.assert_allclose(permissive.values, [3.0, 3.0])

    forecast = xr.DataArray([1.0, 2.0, 3.0], dims="lead_time", coords={"lead_time": [1, 2, 3]})
    accumulated = TimeWindowAccumulator(2).transform(forecast)
    np.testing.assert_array_equal(accumulated.values, [3.0, 5.0])
    np.testing.assert_array_equal(accumulated.lead_time.values, [2, 3])


def test_grid_interpolation_and_matcher_latitude_weights():
    grid = xr.DataArray(
        [[1.0, 3.0], [5.0, 7.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 60.0], "lon": [0.0, 10.0]},
    )
    resized = resize(grid, lat=[0.0, 30.0, 60.0], lon=[0.0, 10.0])
    np.testing.assert_allclose(resized.sel(lat=30.0).values, [3.0, 5.0])
    regridded = RegularLatLonRegridder(
        target_lat=np.array([30.0]), target_lon=np.array([0.0, 10.0])
    ).apply(grid)
    np.testing.assert_allclose(regridded, [[3.0, 5.0]])

    weights = _latitude_weights(grid.to_dataset(name="value"))
    np.testing.assert_allclose(weights.values, [[1.0, 1.0], [0.5, 0.5]])


def test_ensemble_mean_keeps_bundle_metadata_and_missing_policy():
    data = xr.Dataset(
        {"wind": (("member", "lat"), [[1.0, np.nan], [3.0, 5.0]])},
        coords={"member": [0, 1], "lat": [10.0, 20.0]},
    )
    bundle = DataBundle(
        payload=data,
        kind=DataKind.GRIDDED_FORECAST,
        source_id="test",
        standard_vars={"wind": "wind"},
        semantic=SemanticMetadata(
            units={"wind": "m/s"}, temporal_kind=TemporalKind.INSTANTANEOUS, member_count=2
        ),
        provenance=Provenance(input_files=[], reader_id="test", reader_version="1"),
    )
    mean = mean_bundle(bundle)
    np.testing.assert_allclose(mean.payload.wind.values, [2.0, np.nan])
    assert mean.semantic.member_count is None
    assert mean.provenance.transform_chain == ["ensemble_mean_over_member"]
    assert bundle.semantic.member_count == 2
    assert bundle.provenance.transform_chain == []
    np.testing.assert_allclose(mean_bundle(bundle, skipna=True).payload.wind.values, [2.0, 5.0])
    assert mean_bundle(mean) is mean
