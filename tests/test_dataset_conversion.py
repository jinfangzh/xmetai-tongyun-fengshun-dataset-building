import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from pipelines.dataset import extract_zarr_to_netcdf
from pipelines.dataset.old import conversion as _conversion


@pytest.mark.parametrize("dimension", ["channel", "level"])
def test_external_statistic_aligns_labels_and_selects_channels(tmp_path, dimension):
    xr.Dataset(
        {"mean": (dimension, [20.0, 30.0, 10.0])},
        coords={dimension: ["b", "c", "a"]},
    ).to_netcdf(tmp_path / "mean.nc")
    result = _conversion._load_channel_statistic(tmp_path, "mean", ["a", "b"])
    assert result.channel.values.tolist() == ["a", "b"]
    np.testing.assert_allclose(result, [10.0, 20.0])


@pytest.mark.parametrize(
    "labels, message",
    [(["a", "a"], "unique"), (["a", "c"], "missing requested channels")],
)
def test_external_statistic_rejects_ambiguous_or_missing_labels(tmp_path, labels, message):
    xr.Dataset({"std": ("channel", [1.0, 2.0])}, coords={"channel": labels}).to_netcdf(
        tmp_path / "std.nc"
    )
    with pytest.raises(ValueError, match=message):
        _conversion._load_channel_statistic(tmp_path, "std", ["a", "b"])


def test_unlabeled_statistics_require_matching_length(tmp_path):
    np.save(tmp_path / "mean.npy", [10.0, 20.0])
    result = _conversion._load_channel_statistic(tmp_path, "mean", ["a", "b"])
    assert result.channel.values.tolist() == ["a", "b"]
    np.testing.assert_allclose(result, [10.0, 20.0])
    with pytest.raises(ValueError, match="length"):
        _conversion._load_channel_statistic(tmp_path, "mean", ["a"])


@pytest.mark.parametrize("label_dtype", ["U1", "S1"])
def test_native_zarr_statistic_preserves_channel_labels(tmp_path, monkeypatch, label_dtype):
    # Exercise native array handling without requiring a Zarr runtime/store.
    (tmp_path / ".zgroup").write_text("{}")
    (tmp_path / "mean").mkdir()
    (tmp_path / "channel").mkdir()
    arrays = {
        str(tmp_path / "mean"): np.array([20.0, 10.0]),
        str(tmp_path / "channel"): np.array(["b", "a"], dtype=label_dtype),
    }
    monkeypatch.setitem(
        sys.modules,
        "zarr",
        SimpleNamespace(open=lambda path, mode: arrays[path], Group=type("Group", (), {})),
    )
    result = _conversion._load_channel_statistic(tmp_path, "mean", ["a", "b"])
    np.testing.assert_allclose(result, [10.0, 20.0])


def test_export_denormalizes_using_external_channel_labels(tmp_path, monkeypatch):
    stats = tmp_path / "stats"
    stats.mkdir()
    for name, values in (("mean", [20.0, 10.0]), ("std", [2.0, 1.0])):
        xr.Dataset({name: ("channel", values)}, coords={"channel": ["b", "a"]}).to_netcdf(
            stats / f"{name}.nc"
        )
    source = xr.Dataset(
        {"data": (("time", "channel"), [[1.0, 1.0]])},
        coords={"time": [np.datetime64("2025-01-01")], "channel": ["a", "b"]},
    )
    source_path = tmp_path / "input.zarr"
    raw_open = _conversion.dataset_io.open
    monkeypatch.setattr(
        _conversion.dataset_io,
        "open",
        lambda path: (
            SimpleNamespace(read=lambda: source) if path == source_path else raw_open(path)
        ),
    )
    output = extract_zarr_to_netcdf(
        source_path,
        tmp_path / "result.nc",
        start_date="2025-01-01",
        stats_path=stats,
    )
    with xr.open_dataset(output) as result:
        np.testing.assert_allclose(result.data, [[11.0, 22.0]])
