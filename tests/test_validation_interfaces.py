"""Result-style array checks and their callers stop safely on invalid input."""

import numpy as np
import pytest
import xarray as xr

from datakit.validation.check_result import CheckResult
from datakit.validation.structure import (
    check_channels,
    check_coordinates,
    check_dimensions,
    check_same_layout,
)


def weather_array():
    return xr.DataArray(
        np.arange(16, dtype="f4").reshape(2, 2, 2, 2),
        dims=("time", "channel", "lat", "lon"),
        coords={
            "time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
            "channel": ["a", "b"],
            "lat": [90.0, 89.0],
            "lon": [0.0, 1.0],
        },
        name="data",
    )


def test_result_checks_preserve_input_and_do_not_read_main_values(monkeypatch):
    array = weather_array()
    before = array.copy(deep=True)
    read_values = xr.DataArray.values.fget

    def coordinates_only(value):
        if value.name == "data":
            raise AssertionError("Structure checks must not read main data")
        return read_values(value)

    with monkeypatch.context() as patch:
        patch.setattr(xr.DataArray, "values", property(coordinates_only))
        checks = check_dimensions(array) + check_coordinates(array) + check_channels(array)
        checks += check_same_layout(array, array)
    assert all(isinstance(check, CheckResult) and check.status == "pass" for check in checks)
    xr.testing.assert_identical(array, before)


@pytest.mark.parametrize(
    "fault, failed_id",
    [
        ("missing_dimension", "dimensions.lon.present"),
        ("empty_dimension", "dimensions.lat.nonempty"),
        ("missing_coordinate", "coordinates.lat.present"),
        ("duplicate_coordinate", "coordinates.lat.unique"),
        ("descending_time", "coordinates.time.increasing"),
        ("nat", "coordinates.time.valid"),
        ("invalid_time", "coordinates.time.valid"),
        ("mixed_time", "coordinates.time.valid"),
        ("duplicate_time", "coordinates.time.unique"),
        ("duplicate_channel", "channels.channel.unique"),
        ("missing_channel", "channels.channel.present"),
    ],
)
def test_data_defects_return_failures(fault, failed_id):
    array = weather_array()
    if fault == "missing_dimension":
        array = array.isel(lon=0, drop=True)
    elif fault == "empty_dimension":
        array = array.isel(lat=slice(0, 0))
    elif fault == "missing_coordinate":
        array = array.drop_vars("lat")
    elif fault == "duplicate_coordinate":
        array = array.assign_coords(lat=[90.0, 90.0])
    elif fault == "descending_time":
        array = array.isel(time=[1, 0])
    elif fault == "nat":
        array = array.assign_coords(time=np.array(["2025-01-01", "NaT"], dtype="datetime64[ns]"))
    elif fault == "invalid_time":
        array = array.assign_coords(time=["invalid-a", "invalid-b"])
    elif fault == "mixed_time":
        array = array.assign_coords(time=np.array(["2025-01-01", None], dtype=object))
    elif fault == "duplicate_time":
        array = array.isel(time=[0, 0])
    elif fault == "duplicate_channel":
        array = array.assign_coords(channel=["a", "a"])
    else:
        array = array.drop_vars("channel")
    checks = check_dimensions(array) + check_coordinates(array) + check_channels(array)
    assert any(check.check_id == failed_id and check.status == "fail" for check in checks)


@pytest.mark.parametrize(
    "fault, failed_id", [("shape", "layout.shape"), ("order", "layout.dimensions")]
)
def test_layout_mismatch_returns_failures(fault, failed_id):
    left = weather_array()
    right = (
        left.isel(time=slice(0, 1))
        if fault == "shape"
        else left.transpose("channel", "time", "lat", "lon")
    )
    assert any(
        check.check_id == failed_id and check.status == "fail"
        for check in check_same_layout(left, right)
    )


def test_level_channels_can_be_checked_explicitly():
    checks = check_channels(weather_array().rename(channel="level"), coordinate="level")
    assert all(check.status == "pass" for check in checks)


def test_non_vector_coordinate_is_a_failed_result():
    array = weather_array().assign_coords(lat=(("lat", "lon"), [[90.0, 90.0], [89.0, 89.0]]))
    checks = check_coordinates(array)
    assert any(
        check.check_id == "coordinates.lat.dimensions" and check.status == "fail"
        for check in checks
    )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_time_check_results_are_json_serializable_at_all_precisions(unit):
    import json
    from dataclasses import asdict

    from pipelines.dataset.reports import to_jsonable

    array = weather_array().assign_coords(
        time=np.array(["2025-01-01", "2025-01-02"], dtype=f"datetime64[{unit}]"),
    )
    checks = check_coordinates(array)
    assert all(check.status == "pass" for check in checks)
    encoded = json.dumps(to_jsonable([asdict(check) for check in checks]))
    assert "2025-01-01" in encoded


def test_bias_rejects_layout_before_any_value_calculation(monkeypatch):
    from datakit.validation.comparison import compute_bias_mean

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Invalid layout must prevent numerical work")

    left = weather_array()
    with monkeypatch.context() as patch:
        patch.setattr(xr, "apply_ufunc", unexpected_calculation)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_bias_mean(left, left.isel(time=slice(0, 1)))
    assert compute_bias_mean(left + 2, left) == 2


def test_netcdf_example_prints_checks_and_stops_on_invalid_input(monkeypatch, capsys):
    from examples.validation import validate_netcdf

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Invalid structure must prevent diagnostics")

    monkeypatch.setattr(validate_netcdf, "count_invalid", unexpected_calculation)
    monkeypatch.setattr(validate_netcdf, "analyze_range", unexpected_calculation)
    validate_netcdf._print_basic_quality("invalid", weather_array().drop_vars("lat"))
    output = capsys.readouterr().out
    assert "Validation failed" in output
    assert "coordinates.lat.present" in output


def test_old_comparison_saves_failed_checks_without_loading_statistics(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from pipelines.dataset.old import netcdf_comparison

    closed = []
    left = weather_array().isel(lon=0, drop=True).to_dataset()
    right = weather_array().to_dataset()
    left.set_close(lambda: closed.append("left"))
    right.set_close(lambda: closed.append("right"))
    monkeypatch.setattr(
        netcdf_comparison.dataset_io,
        "open",
        lambda path: SimpleNamespace(read=lambda: left if path == "source" else right),
    )

    def unexpected_work(*args, **kwargs):
        raise AssertionError("Invalid input must prevent statistics reading and comparison")

    monkeypatch.setattr(netcdf_comparison, "denormalize_zarr_array", unexpected_work)
    monkeypatch.setattr(netcdf_comparison, "compute_bias_mean", unexpected_work)
    report_path = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="Missing required dimension"):
        netcdf_comparison.validate_zarr_netcdf_consistency(
            "source", "target", report_json=report_path
        )
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert any(check["status"] == "fail" for check in report["checks"]["zarr"])
    assert set(closed) == {"left", "right"}


def test_old_comparison_keeps_bias_statistics_and_json_report(tmp_path):
    import json

    from pipelines.dataset import validate_zarr_netcdf_consistency

    data = weather_array()
    mean = xr.DataArray(
        np.array([10, 20], dtype="f4"), dims="channel", coords={"channel": data.channel}
    )
    std = xr.DataArray(
        np.array([2, 3], dtype="f4"), dims="channel", coords={"channel": data.channel}
    )
    source, target = tmp_path / "source.zarr", tmp_path / "target.nc"
    xr.Dataset({"data": data, "mean": mean, "std": std}).to_zarr(source)
    (data * std + mean).to_dataset(name="data").to_netcdf(target)
    output = tmp_path / "passed.json"
    report = validate_zarr_netcdf_consistency(source, target, report_json=output)
    assert report["status"] == "passed"
    assert report["elementwise"]["bias_mean"] == 0
    assert report["elementwise"]["bias_mean_by_channel"] == {"a": 0, "b": 0}
    assert report["statistics"]["diff_netcdf_minus_zarr_denorm"]["mean"] == 0
    assert json.loads(output.read_text()) == report


def test_retired_validation_interfaces_are_not_exposed():
    from datakit import validation
    from datakit.validation import comparison, structure

    names = (
        "validate_structure",
        "validate_coordinate_values",
        "validate_channel_values",
        "validate_same_layout",
        "validate_weather_array",
    )
    for module in (validation, comparison, structure):
        assert all(not hasattr(module, name) for name in names)
    assert not set(names) & set(validation.__all__)
    for name in ("check_dimensions", "check_coordinates", "check_channels", "check_same_layout"):
        assert callable(getattr(validation, name))


def test_legacy_file_helpers_live_only_with_legacy_workflow(tmp_path):
    from datakit import validation
    from datakit.validation import comparison
    from pipelines.dataset.old import netcdf_comparison

    for name in ("load_channel_statistic_from_zarr", "denormalize_zarr_array", "save_report_json"):
        assert not hasattr(validation, name)
        assert not hasattr(comparison, name)
        assert name not in validation.__all__
        assert getattr(netcdf_comparison, name).__module__ == netcdf_comparison.__name__
    stats = xr.Dataset(
        {
            "mean": ("channel", np.array([10, 20], dtype="f4")),
            "std": ("channel", np.array([2, 3], dtype="f4")),
        },
        coords={"channel": ["a", "b"]},
    )
    path = tmp_path / "statistics.zarr"
    stats.to_zarr(path)
    normalized = xr.DataArray([1, 1], dims="channel", coords={"channel": ["a", "b"]})
    actual = netcdf_comparison.denormalize_zarr_array(normalized, path)
    np.testing.assert_array_equal(actual, [12, 23])
    with pytest.raises(ValueError, match="length"):
        netcdf_comparison.load_channel_statistic_from_zarr(path, "mean", ["a"])
    with pytest.raises(FileNotFoundError):
        netcdf_comparison.load_channel_statistic_from_zarr(path, "missing", ["a", "b"])
