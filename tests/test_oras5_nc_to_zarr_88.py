from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

spec = importlib.util.spec_from_file_location(
    "oras5_nc_to_zarr_88",
    Path(__file__).parents[1] / "examples/dataset_building/oras5_nc_to_zarr_88.py",
)
assert spec is not None and spec.loader is not None
oras5_nc_to_zarr_88 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oras5_nc_to_zarr_88)


def _write_flat_daily(
    path: Path,
    channels: list[str],
    values: np.ndarray,
    with_time: bool = True,
    var_name: str = "data",
) -> None:
    dims = ("time", "channel", "lat", "lon") if with_time else ("channel", "lat", "lon")
    arr = values.astype("float32")[None, ...] if with_time else values.astype("float32")
    coords = {"channel": channels, "lat": [10.0], "lon": [20.0]}
    if with_time:
        coords["time"] = [pd.Timestamp("2025-01-01")]
    xr.Dataset({var_name: (dims, arr)}, coords=coords).to_netcdf(path)


def _write_stats(mean_path: Path, std_path: Path, mask_path: Path, channels: list[str]) -> None:
    mean_values = {name: float(i + 1) for i, name in enumerate(channels)}
    std_values = {name: float((i + 1) * 2) for i, name in enumerate(channels)}
    xr.Dataset(
        {"mean": ("channel", [mean_values[c] for c in channels])},
        coords={"channel": channels},
    ).to_netcdf(mean_path)
    xr.Dataset(
        {"std": ("channel", [std_values[c] for c in channels])},
        coords={"channel": channels},
    ).to_netcdf(std_path)
    xr.Dataset(
        {"mask": (("mask_channel", "lat", "lon"), [[[1.0]]])},
        coords={"mask_channel": ["mask"], "lat": [10.0], "lon": [20.0]},
    ).to_netcdf(mask_path)


def _patch_args(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    destination: Path,
    mean: Path,
    std: Path,
    mask: Path,
    channels: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    strict: bool = False,
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
            "start_date": start_date,
            "end_date": end_date,
            "channels": channels,
            "dataset_id": "test-oras5",
            "schema_version": "1.0",
            "content_version": "v1",
            "data_revision": "test",
            "strict": strict,
        },
    )()
    monkeypatch.setattr(oras5_nc_to_zarr_88, "get_args", lambda: args)




@pytest.mark.parametrize(
    "value, expected",
    [
        ("20250101", pd.Timestamp("2025-01-01").date()),
        ("2025-01-01", pd.Timestamp("2025-01-01").date()),
        (None, None),
    ],
)
def test_parse_cli_date(value: str | None, expected) -> None:
    assert oras5_nc_to_zarr_88._parse_cli_date(value) == expected


def test_infer_date_range_from_flat_files(tmp_path: Path) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    channels = ["so50"]
    _write_flat_daily(source / "20250101_oras5.nc", channels, np.array([[[5.0]]]))
    _write_flat_daily(source / "20250103_oras5.nc", channels, np.array([[[6.0]]]))

    start, end = oras5_nc_to_zarr_88._infer_date_range(source, channels)
    assert start == pd.Timestamp("2025-01-01").date()
    assert end == pd.Timestamp("2025-01-03").date()


def test_infer_available_channels_uses_stats_intersection(tmp_path: Path) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"

    # data has 3 channels, stats only keep 2 so intersection should be returned
    data_channels = ["so50", "to50", "sit"]
    stat_channels = ["to50", "sit"]
    _write_flat_daily(
        source / "20250101_oras5.nc", data_channels, np.array([[[3.0]], [[5.0]], [[7.0]]])
    )
    _write_stats(mean_path, std_path, mask_path, stat_channels)

    mean = oras5_nc_to_zarr_88._load_stat_array(mean_path, "mean")
    std = oras5_nc_to_zarr_88._load_stat_array(std_path, "std")

    channels = oras5_nc_to_zarr_88._infer_available_channels(source, mean, std)
    assert channels == stat_channels


def test_main_flat_daily_to_zarr_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"

    channels = ["so50", "to50"]
    # values shape: (channel, lat, lon)
    _write_flat_daily(
        source / "20250101_oras5.nc", channels, np.array([[[3.0]], [[8.0]]]), with_time=False
    )
    _write_flat_daily(
        source / "20250102_oras5.nc", channels, np.array([[[5.0]], [[10.0]]]), with_time=False
    )
    _write_stats(mean_path, std_path, mask_path, channels)

    _patch_args(
        monkeypatch,
        source,
        destination,
        mean_path,
        std_path,
        mask_path,
        channels="",
        start_date=None,
        end_date=None,
    )

    oras5_nc_to_zarr_88.main()

    ds = xr.open_zarr(destination, consolidated=False)
    try:
        assert ds.sizes["time"] == 2
        assert ds.sizes["channel"] == 2
        assert ds.channel.values.tolist() == channels
        assert "data" in ds and "mean" in ds and "std" in ds and "mask" in ds
        assert ds["data"].dtype == np.dtype("float16")

        # mean=[1,2], std=[2,4]
        expected = np.array(
            [
                [[[1.0]], [[1.5]]],
                [[[2.0]], [[2.0]]],
            ],
            dtype="float32",
        )
        np.testing.assert_allclose(
            ds["data"].values.astype("float32"), expected, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(ds["mean"].values, [1.0, 2.0])
        np.testing.assert_allclose(ds["std"].values, [2.0, 4.0])
        assert ds["mask"].dims == ("mask_channel", "lat", "lon")
        assert ds.attrs["dataset_id"] == "test-oras5"
    finally:
        ds.close()


def test_main_channel_stat_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"

    _write_flat_daily(source / "20250101_oras5.nc", ["so50"], np.array([[[3.0]]]))
    _write_stats(mean_path, std_path, mask_path, ["to50"])

    _patch_args(
        monkeypatch,
        source,
        destination,
        mean_path,
        std_path,
        mask_path,
        channels="so50",
        start_date="2025-01-01",
        end_date="2025-01-01",
    )

    with pytest.raises(ValueError, match="channel-stat mismatch"):
        oras5_nc_to_zarr_88.main()


def test_main_missing_channel_in_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "nc"
    source.mkdir()
    destination = tmp_path / "output.zarr"
    mean_path, std_path, mask_path = tmp_path / "mean.nc", tmp_path / "std.nc", tmp_path / "mask.nc"

    _write_flat_daily(source / "20250101_oras5.nc", ["so50"], np.array([[[3.0]]]))
    _write_stats(mean_path, std_path, mask_path, ["so50", "to50"])

    _patch_args(
        monkeypatch,
        source,
        destination,
        mean_path,
        std_path,
        mask_path,
        channels="so50,to50",
        start_date="2025-01-01",
        end_date="2025-01-01",
        strict=True,
    )

    with pytest.raises(KeyError, match=r"missing channel\(s\)"):
        oras5_nc_to_zarr_88.main()
