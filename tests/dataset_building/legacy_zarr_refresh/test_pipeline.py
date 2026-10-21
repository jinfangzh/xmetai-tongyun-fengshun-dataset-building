from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr
from legacy_zarr_refresh import (
    LegacyZarrSource,
    RefreshProfile,
    load_profiles,
    read_manifest,
    refresh_legacy_zarr,
    select_profiles,
)
from legacy_zarr_refresh.source import correct_statistics
from legacy_zarr_refresh.store import (
    create_refreshed_store,
)


def _write_legacy_store(path: Path) -> np.ndarray:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    values = np.arange(16, dtype="f4").reshape(2, 2, 2, 2)
    group.create_array("data", data=values, chunks=(1, 2, 2, 2))
    group.create_array(
        "channel",
        data=np.asarray(["ssrd", "q1000"], dtype=str),
        chunks=(2,),
    )
    group.create_array("lat", data=np.asarray([-1.0, 1.0]), chunks=(2,))
    group.create_array("lon", data=np.asarray([0.0, 180.0]), chunks=(2,))
    time = group.create_array("time", data=np.asarray([0, 6], dtype="i8"), chunks=(2,))
    time.attrs["units"] = "hours since 2024-01-01 00:00:00"
    np.save(path / "mean.npy", np.asarray([4.0, 2.0], dtype="f4"))
    np.save(path / "std.npy", np.asarray([2.0, 0.5], dtype="f4"))
    np.save(
        path / "mask.npy",
        np.asarray([[0.0, 0.8], [1.0, 0.2]], dtype="f4"),
    )
    return values


def _profile(source: Path) -> RefreshProfile:
    return RefreshProfile(
        key="synthetic",
        input=source,
        output_name="synthetic.refreshed.zarr",
        cadence_hours=6,
        radiation_seconds=21600,
        statistics_divide={"q1000": 2.0},
        statistics_multiply={"ssrd": 3.0},
        mask={"kind": "npy", "path": "mask.npy", "threshold": 0.5},
        expected_channels=2,
        expected_grid=(2, 2),
    )


def test_profiles_are_external_and_all_selects_every_profile(tmp_path: Path) -> None:
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps(
            {
                "datasets": {
                    "sample": {
                        "input": "/external/sample.zarr",
                        "output_name": "sample.refreshed.zarr",
                        "cadence_hours": 6,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    profiles = load_profiles(config)
    assert [profile.key for profile in select_profiles(profiles, ["all"])] == ["sample"]
    assert profiles["sample"].already_refreshed == "error"


def test_statistics_corrections_are_channel_aligned() -> None:
    mean, std = correct_statistics(
        ("q1000", "ssrd"),
        np.asarray([2.0, 4.0], dtype="f4"),
        np.asarray([0.5, 2.0], dtype="f4"),
        {"q1000": 2.0},
        {"ss*": 3.0},
    )
    np.testing.assert_allclose(mean, [1.0, 12.0])
    np.testing.assert_allclose(std, [0.25, 6.0])


def test_statistics_factor_one_is_a_no_op() -> None:
    original_mean = np.asarray([2.0, 4.0], dtype="f4")
    original_std = np.asarray([0.5, 2.0], dtype="f4")
    mean, std = correct_statistics(
        ("q1000", "ssrd"),
        original_mean,
        original_std,
        {"q1000": 1.0},
        {"ss*": 1.0},
    )
    np.testing.assert_array_equal(mean, original_mean)
    np.testing.assert_array_equal(std, original_std)


def test_already_refreshed_source_is_rejected_before_open(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    (source / "refresh_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains a refresh manifest"):
        LegacyZarrSource(_profile(source))


def test_refresh_store_uses_zarr_v2_and_dot_chunk_keys(tmp_path: Path) -> None:
    store = tmp_path / "refreshed.zarr"
    create_refreshed_store(
        store,
        dataset_id="synthetic",
        data_revision="test",
        times=np.asarray(["2024-01-01"], dtype="datetime64[D]"),
        channels=("t2m", "q1000"),
        lat=np.asarray([1.0, -1.0]),
        lon=np.asarray([0.0, 180.0]),
        mean=np.asarray([0.0, 0.0]),
        std=np.asarray([1.0, 1.0]),
        land_mask=np.asarray([[True, False], [False, True]]),
        radiation_seconds=None,
    )
    zarr.consolidate_metadata(str(store))

    root = json.loads((store / ".zgroup").read_text(encoding="utf-8"))
    data = json.loads((store / "data" / ".zarray").read_text(encoding="utf-8"))
    assert root["zarr_format"] == 2
    assert data["dimension_separator"] == "."
    assert (store / ".zmetadata").is_file()


def test_invalid_statistics_pattern_is_rejected() -> None:
    with pytest.raises(ValueError, match="matched no channels"):
        correct_statistics(("t2m",), np.asarray([0.0]), np.asarray([1.0]), {}, {"q*": 2.0})


def test_end_to_end_refresh_aligns_channels_latitude_statistics_and_mask(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "legacy.zarr"
    source_values = _write_legacy_store(source_path)
    output_root = tmp_path / "output"

    output, manifest = refresh_legacy_zarr(
        _profile(source_path),
        output_root,
        full_data_check=True,
    )

    group = zarr.open_group(str(output), mode="r")
    assert group["channel"][:].tolist() == ["q1000", "ssrd"]
    np.testing.assert_array_equal(group["lat"][:], [1.0, -1.0])
    np.testing.assert_array_equal(group["lon"][:], [0.0, 180.0])
    np.testing.assert_allclose(group["mean"][:], [1.0, 12.0])
    np.testing.assert_allclose(group["std"][:], [0.25, 6.0])

    expected = source_values[:, [1, 0], ::-1, :].astype("f2")
    np.testing.assert_array_equal(group["data"][:], expected)
    np.testing.assert_array_equal(
        group["mask"][0],
        np.asarray([[1, 0], [0, 1]], dtype="u1"),
    )
    assert manifest["validation"]["status"] == "ok"
    assert manifest["validation"]["full_data_check"] is True
    assert len(manifest["validation"]["checked_data_chunks"]) == 2
    assert (output / "refresh_manifest.json").is_file()
    reopened_manifest = read_manifest(output)
    assert reopened_manifest["validation"]["status"] == "ok"
    assert reopened_manifest["source_indices"] == {
        "start": 0,
        "end": 1,
        "count": 2,
    }


def external_store(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to run this external-data check")
    return Path(value)


def assert_dataset_store(path: Path) -> None:
    assert (path / ".zgroup").is_file(), f"not a Zarr v2 store: {path}"
    group = zarr.open_group(str(path), mode="r")
    assert {
        "data",
        "time",
        "channel",
        "lat",
        "lon",
        "mean",
        "std",
        "mask",
        "mask_channel",
    } <= set(group)
    assert group["data"].ndim == 4
    assert group["data"].shape[1] == group["channel"].shape[0]
    assert group["mean"].shape == group["std"].shape == group["channel"].shape
    assert group["mask"].shape[1:] == group["data"].shape[2:]


@pytest.mark.integration
def test_external_refreshed_store_contract() -> None:
    assert_dataset_store(external_store("XMETAI_REFRESH_TEST_STORE"))
