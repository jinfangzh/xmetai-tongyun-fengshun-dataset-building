from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

spec = importlib.util.spec_from_file_location(
    "era5_nc_to_zarr", Path(__file__).parents[1] / "examples/dataset_building/era5_nc_to_zarr.py"
)
assert spec is not None and spec.loader is not None
era5_nc_to_zarr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(era5_nc_to_zarr)


def _write_nc(
    path: Path,
    channels: list[str],
    values: np.ndarray,
    time: str,
    var_name: str = "data",
    with_time: bool = True,
) -> None:
    dims = ("time", "channel", "lat", "lon") if with_time else ("channel", "lat", "lon")
    arr = values.astype("float32")[None, ...] if with_time else values.astype("float32")
    coords = {"channel": channels, "lat": [90.0], "lon": [0.0]}
    if with_time:
        coords["time"] = [pd.Timestamp(time)]
    ds = xr.Dataset({var_name: (dims, arr)}, coords=coords)
    ds.to_netcdf(path)


def _write_stats(mean_path: Path, std_path: Path, mask_path: Path, channels: list[str]) -> None:
    mean_values = {"q1000": 10.0, "ssrd": 20.0, "z1000": 30.0, "tp": 1.0}
    std_values = {"q1000": 2.0, "ssrd": 4.0, "z1000": 5.0, "tp": 0.5}
    xr.Dataset(
        {"mean": ("channel", [mean_values[c] for c in channels])}, coords={"channel": channels}
    ).to_netcdf(mean_path)
    xr.Dataset(
        {"std": ("channel", [std_values[c] for c in channels])}, coords={"channel": channels}
    ).to_netcdf(std_path)
    xr.Dataset(
        {"mask": (("mask_channel", "lat", "lon"), [[[1.0]]])},
        coords={"mask_channel": ["mask"], "lat": [90.0], "lon": [0.0]},
    ).to_netcdf(mask_path)


def _patch_args(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    destination: Path,
    mean: Path,
    std: Path,
    mask: Path,
    channels: str = "",
    accumulation_hours: float = 24.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    args = type(
        "Args",
        (),
        {
            "source": source,
            "destination": destination,
            "mean_nc": mean,
            "std_nc": std,
            "mask_nc": mask,
            "accumulation_hours": accumulation_hours,
            "start_date": start_date,
            "end_date": end_date,
            "channels": channels,
            "dataset_id": "test-era5",
            "schema_version": "1.0",
            "content_version": "v1",
            "data_revision": "test",
        },
    )()
    monkeypatch.setattr(era5_nc_to_zarr, "get_args", lambda: args)




def test_nc_to_zarr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000", "ssrd", "z1000", "tp"]

    _write_nc(
        source / "foo.nc",
        channels,
        np.array([[[12.0]], [[3600.0 * 24.0 * 2.0]], [[34.0]], [[0.001]]]),
        "2025-01-01",
    )
    _write_nc(
        source / "random_name.nc",
        channels,
        np.array([[[14.0]], [[3600.0 * 24.0 * 3.0]], [[38.0]], [[0.003]]]),
        "2025-01-02",
    )
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(monkeypatch, source, destination, mean_path, std_path, mask_path)

    era5_nc_to_zarr.main()

    ds = xr.open_zarr(destination, consolidated=False)
    try:
        assert ds.sizes["time"] == 2
        assert ds.sizes["channel"] == 4
        assert ds.channel.values.tolist() == channels
        assert "data" in ds and "mean" in ds and "std" in ds and "mask" in ds
        assert ds["data"].dtype == np.dtype("float16")

        expected_tp_1 = (np.log1p(1.0) - 1.0) / 0.5
        expected_tp_2 = (np.log1p(3.0) - 1.0) / 0.5
        expected = np.array(
            [
                [[[5995.0]], [[-4.5]], [[0.8]], [[expected_tp_1]]],
                [[[6995.0]], [[-4.25]], [[1.6]], [[expected_tp_2]]],
            ],
            dtype="float32",
        )

        np.testing.assert_allclose(ds["data"].values, expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ds["mean"].values, [10.0, 20.0, 30.0, 1.0])
        np.testing.assert_allclose(ds["std"].values, [2.0, 4.0, 5.0, 0.5])
        assert ds["mask"].dims == ("mask_channel", "lat", "lon")
        np.testing.assert_allclose(ds["mask"].values, [[[1.0]]])
        assert ds.attrs["dataset_id"] == "test-era5"
        assert ds.attrs["schema_version"] == "1.0"
        assert ds.attrs["content_version"] == "v1"
        assert ds.attrs["data_revision"] == "test"
    finally:
        ds.close()


def test_nc_to_zarr_6h_radiation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["ssrd"]

    _write_nc(source / "data.nc", channels, np.array([[[3600.0 * 6.0 * 2.0]]]), "2025-01-01")
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(
        monkeypatch, source, destination, mean_path, std_path, mask_path, accumulation_hours=6.0
    )

    era5_nc_to_zarr.main()

    ds = xr.open_zarr(destination, consolidated=False)
    try:
        assert ds.sizes["time"] == 1
        assert ds.sizes["channel"] == 1
        assert ds.channel.values.tolist() == ["ssrd"]
        expected = (2.0 - 20.0) / 4.0
        np.testing.assert_allclose(ds["data"].values, [[[[expected]]]], rtol=1e-3, atol=1e-3)
    finally:
        ds.close()


def test_nc_to_zarr_with_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000", "ssrd", "z1000"]

    for filename, day, value in [
        ("abc.nc", "2025-01-01", 10.0),
        ("random_file.nc", "2025-01-02", 20.0),
        ("something_else.nc", "2025-01-03", 30.0),
    ]:
        _write_nc(
            source / filename, channels, np.array([[[value]], [[3600.0 * 24.0]], [[30.0]]]), day
        )

    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(
        monkeypatch,
        source,
        destination,
        mean_path,
        std_path,
        mask_path,
        channels="q1000,z1000",
        start_date="2025-01-02",
        end_date="2025-01-03",
    )

    era5_nc_to_zarr.main()

    ds = xr.open_zarr(destination, consolidated=False)
    try:
        assert ds.sizes["time"] == 2
        assert ds.sizes["channel"] == 2
        assert ds.channel.values.tolist() == ["q1000", "z1000"]
        np.testing.assert_array_equal(
            ds.time.values, [np.datetime64("2025-01-02"), np.datetime64("2025-01-03")]
        )
        np.testing.assert_allclose(ds["mean"].values, [10.0, 30.0])
        np.testing.assert_allclose(ds["std"].values, [2.0, 5.0])
    finally:
        ds.close()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("20250101", pd.Timestamp("2025-01-01").date()),
        ("2025-01-01", pd.Timestamp("2025-01-01").date()),
        (None, None),
    ],
)
def test_parse_cli_date(value: str | None, expected) -> None:
    assert era5_nc_to_zarr._parse_cli_date(value) == expected


def test_main_no_nc_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000"]
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(monkeypatch, source, destination, mean_path, std_path, mask_path)

    with pytest.raises(FileNotFoundError, match="没有找到 NetCDF 文件"):
        era5_nc_to_zarr.main()


def test_main_missing_data_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000"]

    _write_nc(source / "bad.nc", channels, np.array([[[1.0]]]), "2025-01-01", var_name="other")
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(monkeypatch, source, destination, mean_path, std_path, mask_path)

    with pytest.raises(KeyError, match="未找到变量 'data'"):
        era5_nc_to_zarr.main()


def test_main_missing_time_dimension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000"]

    _write_nc(source / "bad.nc", channels, np.array([[[1.0]]]), "2025-01-01", with_time=False)
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(monkeypatch, source, destination, mean_path, std_path, mask_path)

    with pytest.raises(ValueError, match="必须包含 'time' dimension"):
        era5_nc_to_zarr.main()


def test_main_channels_missing_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000", "z1000"]

    _write_nc(source / "a.nc", channels, np.array([[[10.0]], [[30.0]]]), "2025-01-01")
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(
        monkeypatch, source, destination, mean_path, std_path, mask_path, channels="q1000,tp"
    )

    with pytest.raises(KeyError, match="缺少指定 channels"):
        era5_nc_to_zarr.main()


def test_main_channel_stats_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"

    _write_nc(source / "a.nc", ["q1000", "z1000"], np.array([[[10.0]], [[30.0]]]), "2025-01-01")
    _write_stats(mean_path, std_path, mask_path, ["q1000"])
    _patch_args(monkeypatch, source, destination, mean_path, std_path, mask_path)

    with pytest.raises(ValueError, match="channel 与统计量不匹配"):
        era5_nc_to_zarr.main()


def test_main_no_data_in_date_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"
    channels = ["q1000"]

    _write_nc(source / "a.nc", channels, np.array([[[10.0]]]), "2025-01-01")
    _write_stats(mean_path, std_path, mask_path, channels)
    _patch_args(
        monkeypatch,
        source,
        destination,
        mean_path,
        std_path,
        mask_path,
        start_date="2025-02-01",
        end_date="2025-02-02",
    )

    with pytest.raises(FileNotFoundError, match="指定日期范围内没有找到 NetCDF 数据"):
        era5_nc_to_zarr.main()


def test_load_stat_array_requires_channel_dim(tmp_path: Path) -> None:
    path = tmp_path / "mean_bad.nc"
    xr.Dataset({"mean": ("x", [1.0, 2.0])}, coords={"x": [0, 1]}).to_netcdf(path)

    with pytest.raises(ValueError, match="must contain a 'channel' dimension"):
        era5_nc_to_zarr._load_stat_array(path, "mean")
