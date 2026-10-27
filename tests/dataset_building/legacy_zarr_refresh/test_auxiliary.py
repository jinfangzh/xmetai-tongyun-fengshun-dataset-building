from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from legacy_zarr_refresh import (
    build_const,
    create_refreshed_store,
    migrate_auxiliary_files,
)

from datakit.schema.datasets import load_schema
from tests.dataset_building.samples.legacy import profile
from tests.dataset_building.samples.legacy import write_const as _write_const


def _target_store(path: Path) -> None:
    create_refreshed_store(
        path,
        schema=load_schema(profile(path).target_schema)[0],
        times=np.asarray(["2024-01-01"], dtype="datetime64[D]"),
        channels=("q1000", "ssrd"),
        lat=np.asarray([1.0, -1.0]),
        lon=np.asarray([0.0, 2.0]),
        mean=np.asarray([0.0, 0.0]),
        std=np.asarray([1.0, 1.0]),
        land_mask=np.asarray([[True, False], [False, True]]),
    )


def test_const_refresh_preserves_legacy_channels_without_inventing_six(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.zarr"
    _target_store(target)
    source = tmp_path / "const.nc"
    _write_const(source)

    refreshed, details = build_const(source, target)
    try:
        assert refreshed.channel.values.tolist() == ["z", "lsm"]
        assert details["generated_channels"] == []
        assert details["channel_policy"] == "preserved from legacy const.nc"
        assert details["latitude_reordered"] is True
        assert details["longitude_reordered"] is True
    finally:
        refreshed.close()


def test_auxiliary_migration_aligns_const_weight_and_mask(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    _write_const(source / "const.nc")
    xr.Dataset(
        {"weight": (("lat",), np.asarray([10.0, 20.0], dtype="f4"))},
        coords={"lat": np.asarray([-1.0, 1.0])},
    ).to_netcdf(source / "weight.nc")
    xr.Dataset(
        {"lsm": (("lat", "lon"), np.asarray([[0, 1], [1, 1]], dtype="u1"))},
        coords={
            "lat": np.asarray([-1.0, 1.0]),
            "lon": np.asarray([2.0, 0.0]),
        },
    ).to_netcdf(source / "mask.nc")

    target = tmp_path / "synthetic.refreshed.zarr"
    _target_store(target)
    zarr.consolidate_metadata(str(target))

    report = migrate_auxiliary_files(
        "synthetic",
        {
            "source": str(source),
            "outputs": [target.name],
            "refresh_embedded_mask": True,
        },
        tmp_path,
        dry_run=False,
        overwrite=False,
        strict=True,
    )

    assert report["status"] == "ok"
    assert (target / "const.nc").is_file()
    assert (target / "weight.npy").is_file()
    assert (target / "mask.npy").is_file()
    assert target.with_name(target.name + ".auxiliary_refresh_manifest.json").is_file()
    np.testing.assert_array_equal(
        np.load(target / "weight.npy"),
        np.asarray([[[20.0], [10.0]]], dtype="f4"),
    )
    np.testing.assert_array_equal(
        np.load(target / "mask.npy"),
        np.asarray([[1, 1], [1, 0]], dtype=bool),
    )
    assert report["artifacts"]["const.nc"]["generated_channels"] == []
    assert report["artifacts"]["mask.npy"]["source_format"] == "netcdf"
    assert report["artifacts"]["embedded_mask"]["verified"] is True


def test_auxiliary_default_preserves_embedded_mask(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    _write_const(source / "const.nc")
    xr.Dataset(
        {"lsm": (("lat", "lon"), np.asarray([[0, 1], [1, 1]], dtype="u1"))},
        coords={
            "lat": np.asarray([-1.0, 1.0]),
            "lon": np.asarray([2.0, 0.0]),
        },
    ).to_netcdf(source / "mask.nc")

    target = tmp_path / "synthetic.refreshed.zarr"
    _target_store(target)
    before = np.asarray(zarr.open_array(str(target / "mask"), mode="r")[:])

    report = migrate_auxiliary_files(
        "synthetic",
        {"source": str(source), "outputs": [target.name]},
        tmp_path,
        dry_run=False,
        overwrite=False,
        strict=True,
    )

    after = np.asarray(zarr.open_array(str(target / "mask"), mode="r")[:])
    np.testing.assert_array_equal(after, before)
    assert report["artifacts"]["embedded_mask"]["status"] == "preserved"
