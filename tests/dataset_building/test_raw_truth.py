"""Product-specific source selection and checks live with the extraction code."""

from datetime import date

import numpy as np
import pytest
import xarray as xr

from pipelines.dataset.old import raw_truth as _raw_truth


def test_open_selected_reads_only_the_requested_day(tmp_path):
    path = tmp_path / "monthly.nc"
    times = np.arange("2024-02-28T00", "2024-03-02T00", dtype="datetime64[6h]").astype("datetime64[ns]")
    values = np.arange(len(times), dtype="f4").reshape(-1, 1, 1)
    source = xr.Dataset(
        {"u10": (("valid_time", "latitude", "longitude"), values)},
        coords={"valid_time": times, "latitude": [90.0], "longitude": [0.0]},
    )
    source.to_netcdf(path)
    selected = _raw_truth._open_selected(path, "u10m", date(2024, 2, 29))
    assert selected.u10m.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(selected.u10m.values, values[4:8])
    np.testing.assert_array_equal(
        selected.time.values,
        np.array(["2024-02-29T00", "2024-02-29T06", "2024-02-29T12", "2024-02-29T18"], dtype="datetime64[ns]"),
    )
    _raw_truth._validate_six_hourly_day(selected, date(2024, 2, 29))


@pytest.mark.parametrize("hours", [[0, 6, 12], [0, 6, 6, 18], [0, 12, 6, 18]])
def test_day_check_rejects_missing_duplicate_or_unordered_slots(hours):
    dataset = xr.Dataset(coords={"time": np.datetime64("2024-02-29") + np.array(hours, dtype="timedelta64[h]")})
    with pytest.raises(ValueError, match="Unexpected day slots"):
        _raw_truth._validate_six_hourly_day(dataset, date(2024, 2, 29))


def test_layout_check_rejects_channel_reordering():
    with pytest.raises(ValueError, match="Channel names/order"):
        _raw_truth._validate_era5_layout(
            channels=["tp", "t2m"], data_shape=(4, 2, 1, 1),
            lat=np.array([90]), lon=np.array([0]),
            expected_channels=["t2m", "tp"], expected_lat=np.array([90]), expected_lon=np.array([0]),
        )
