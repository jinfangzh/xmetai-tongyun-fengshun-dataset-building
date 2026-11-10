"""Daily CDS statistics must never enter Tongyun's hourly accumulation path."""

import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.schema import load_mapping, load_schema
from examples.dataset_building import build_ready_tongyun_days as consumer
from examples.dataset_building import example_build_era5 as era5
from examples.dataset_building import example_build_fengshun_daily as daily
from pipelines.dataset import build_zarr


def write_zip(path, dataset):
    with ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".nc", dataset.to_netcdf(engine="scipy"))


@pytest.fixture
def daily_input(tmp_path):
    description = io.get_description(daily.SOURCE_CONFIG)
    mapping = load_mapping(description)
    day = date(2026, 11, 4)
    root = tmp_path / "raw" / "cds_daily_1p5"
    baseline, _ = load_schema(daily.TARGET_SPEC)
    grid = replace(baseline.variant.grid, latitude_count=2, longitude_count=2)
    product = replace(baseline.product, variants=(replace(baseline.variant, grid=grid, auxiliaries=()),))
    spec = replace(baseline, product=product, first_time=f"{day}T00", last_time=f"{day}T00",
                   directory_name="result.zarr", identity_mapping_evidence="Synthetic daily test")
    lat, lon = grid.coordinates()
    for variable, template in description["layout"]["files"].items():
        declaration = mapping.variables[variable]
        coords = {"time": [np.datetime64(str(day), "ns")], "lat": lat, "lon": lon}
        dims = ["time"]
        if variable in {"z", "t", "u", "v", "q"}:
            levels = [channel.level for channel in baseline.product.channels if channel.variable == variable]
            coords["level"] = ("level", levels, {"units": "hPa"})
            dims.append("level")
        dims += ["lat", "lon"]
        shape = [len(coords[d][1]) if isinstance(coords[d], tuple) else len(coords[d]) for d in dims]
        value = {"q": .002, "tp": .003, "ttr": -86400}.get(variable, 4)
        field = xr.DataArray(np.full(shape, value, dtype="f4"), dims=dims, coords=coords,
                             name=declaration.source, attrs={"units": declaration.source_unit})
        path = root / template.format(day=datetime.combine(day, time()))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_zip(path, field.to_dataset())
    paths = {}
    channels = list(spec.product.channel_names)
    # Nonidentity statistics prove the existing normalization is applied once.
    mean = xr.DataArray(np.full(76, 2, dtype="f4"), dims="channel", coords={"channel": channels})
    std = xr.full_like(mean, 2)
    mean.loc[dict(channel="tp")] = 0
    std.loc[dict(channel="tp")] = 1
    mask = xr.DataArray(np.array([np.ones((2, 2)), np.zeros((2, 2))], dtype="u1"),
                        dims=("mask_channel", "lat", "lon"),
                        coords={"mask_channel": ["land_mask", "sea_mask"], "lat": lat, "lon": lon})
    for name, array in {"mean": mean, "std": std, "mask": mask}.items():
        paths[name] = tmp_path / f"{name}.nc"
        array.rename(name).to_dataset().to_netcdf(paths[name], engine="scipy")
    return root, day, description, spec, paths


def test_daily_means_and_sums_build_normalized_c76_without_boundary(daily_input, tmp_path):
    root, day, description, spec, paths = daily_input
    assert len(daily.required_files(root, day, description)) == 16
    process = partial(era5.process, mean_path=paths["mean"], std_path=paths["std"],
                      mask_path=paths["mask"], source_stage="packed", accumulation_hours=24)
    with daily.read_source(root, description, spec=spec) as packed:
        result = process(packed, description, spec)
        assert result.data.shape == (1, 76, 2, 2)
        assert list(result.channel.values) == list(spec.product.channel_names)
        np.testing.assert_allclose(result.data.sel(channel="tp").compute(), np.log1p(3), rtol=1e-3)
        np.testing.assert_allclose(result.data.sel(channel="ttr").compute(), -1.5, rtol=1e-3)
        np.testing.assert_allclose(result.data.sel(channel="q1000").compute(), 0, atol=1e-3)
        np.testing.assert_allclose(result.data.sel(channel="t2m").compute(), 1, rtol=1e-3)
    output = tmp_path / "result.zarr"
    report = build_zarr(description, spec, root, output, read=daily.read_source, process=process, dry_run=True)
    assert report.coverage["conformance_passed"], report.to_dict()
    assert not output.exists()


@pytest.mark.parametrize("axis", ["time", "lat", "lon"])
def test_reader_rejects_wrong_timestamp_or_native_grid(daily_input, axis):
    root, day, description, spec, _ = daily_input
    for path, _ in daily.required_files(root, day, description):
        with io.open(path).read() as source:
            altered = source.load()
        delta = np.timedelta64(6, "h") if axis == "time" else .25
        altered = altered.assign_coords({axis: altered[axis] + delta})
        write_zip(path, altered)
    with pytest.raises(ValueError, match="UTC daily timestamp|native 1.5 degree grid"):
        daily.read_source(root, description, spec=spec)


def test_part_file_does_not_count_as_completed_input(daily_input):
    root, day, description, _, _ = daily_input
    path, _ = daily.required_files(root, day, description)[0]
    path.rename(path.with_suffix(".zip.part"))
    with pytest.raises(FileNotFoundError, match="incomplete"):
        daily.required_files(root, day, description)


def _config(daily_input, tmp_path):
    root, day, *_ = daily_input
    watermark = tmp_path / "watermark.json"
    watermark.write_text(json.dumps({"latest_complete_day": str(day)}))
    return dict(source=root, download_watermark=watermark, output_root=tmp_path / "products",
                report_dir=tmp_path / "reports",
                **{n: tmp_path / n for n in ("mean_nc", "std_nc", "mask_nc", "const_nc", "weight_npy")})


def test_consumer_watermark_missing_files_and_dry_run(daily_input, tmp_path, monkeypatch):
    config = _config(daily_input, tmp_path)
    day = daily_input[1]
    assert list(consumer.ready_days(config, day - timedelta(days=1), builder=daily)) == []
    assert list(consumer.ready_days(config, day, builder=daily)) == [day]
    calls = []
    monkeypatch.setattr(daily, "main", lambda args: calls.append(args) or 0)
    assert consumer.run(config, builder=daily) == 0
    assert "--dry-run" in calls[0]
    assert not config["output_root"].exists()
    missing, _ = daily.required_files(config["source"], day, daily_input[2])[0]
    missing.unlink()
    assert list(consumer.ready_days(config, day, builder=daily)) == []
    config["download_watermark"].unlink()
    assert consumer.run(config, builder=daily, write=True) == 0
    assert len(calls) == 1


def test_consumer_retries_then_publishes_and_skips(daily_input, tmp_path, monkeypatch):
    config = _config(daily_input, tmp_path)
    attempts = []

    def fake_build(options):
        output = Path(options[options.index("--destination") + 1])
        assert not output.exists()
        attempts.append(output)
        output.mkdir(parents=True)  # Empty test directory, no actual Zarr mutation.
        return 1 if len(attempts) == 1 else 0

    monkeypatch.setattr(daily, "main", fake_build)
    assert consumer.run(config, builder=daily, write=True) == 1
    assert not (config["output_root"] / "_completed").exists()
    assert consumer.run(config, builder=daily, write=True) == 0
    identity, output, marker = consumer.product_paths(config, daily_input[1], builder=daily)
    assert identity == "s2s.20261104.c76.1p5.h24.v1"
    assert output.is_dir() and marker.is_file()
    assert attempts[0].is_dir() and attempts[0] != attempts[1]
    assert json.loads(marker.read_text())["destination"] == str(output)
    monkeypatch.setattr(daily, "required_files", lambda *_: pytest.fail("completed day must be skipped"))
    assert consumer.run(config, builder=daily, write=True) == 0
    assert len(attempts) == 2


def test_daily_cli_isolates_requested_day(daily_input, tmp_path, monkeypatch):
    root, day, description, _, _ = daily_input
    stages = []

    class Report:
        coverage = {"conformance_passed": True}

    def fake_build(**kwargs):
        stage = kwargs["source_path"]
        stages.append(stage)
        assert len(io.open(stage, description).discover(all_files=True)) == 16
        assert len(list(stage.rglob("*.zip"))) == 16
        assert all(p.is_symlink() for p in stage.rglob("*.zip"))
        assert kwargs["dry_run"]
        assert kwargs["target_spec"] == daily.TARGET_SPEC
        assert kwargs["build_overrides"]["first_time"] == f"{day}T00:00"
        assert kwargs["build_overrides"]["last_time"] == f"{day}T00:00"
        return Report()

    monkeypatch.setattr(daily, "build_zarr", fake_build)
    monkeypatch.setattr(daily, "save_build_report", lambda *_: tmp_path / "report.json")
    monkeypatch.setattr(daily, "print_build_summary", lambda *a, **kw: None)
    args = ["--date", str(day), "--source", str(root), "--destination", str(tmp_path / "result.zarr"),
            "--report-dir", str(tmp_path / "reports")]
    for name in ("mean-nc", "std-nc", "mask-nc", "const-nc", "weight-npy"):
        args.extend([f"--{name}", str(tmp_path / name)])
    assert daily.main(args) == 0
    assert not stages[0].exists()
    assert len(list(root.rglob("*.zip"))) == 16
    assert not list(root.rglob("*.nc"))


def test_zip_reader_normalizes_coordinate_order_without_resampling(daily_input):
    root, day, description, spec, _ = daily_input
    for path, _ in daily.required_files(root, day, description):
        with io.open(path).read() as source:
            altered = source.load().sortby("lat")
        # Same points represented as negative longitudes, not a different grid.
        altered = altered.assign_coords(lon=altered.lon - 360).sortby("lon", ascending=False)
        write_zip(path, altered)
    with daily.read_source(root, description, spec=spec) as packed:
        assert packed.data.shape == (1, 76, 2, 2)
        lat, lon = spec.variant.grid.coordinates()
        np.testing.assert_array_equal(packed.lat, lat)
        np.testing.assert_array_equal(packed.lon, lon)


def test_consumer_backfills_in_order_with_limit_and_watermark(daily_input, tmp_path, monkeypatch):
    config = _config(daily_input, tmp_path)
    root, first, description, *_ = daily_input
    # Discovery may encounter gaps and a future day already on disk. Content
    # validation is mocked here; links only provide the expected file names.
    for offset in (2, 4):
        selected = first + timedelta(days=offset)
        for original, relative in daily.required_files(root, first, description):
            target = root / str(relative).replace(first.strftime("%Y%m%d"), selected.strftime("%Y%m%d"))
            target.symlink_to(original)
    latest = first + timedelta(days=2)
    config["download_watermark"].write_text(json.dumps({"latest_complete_day": str(latest)}))
    assert list(consumer.ready_days(config, latest, builder=daily)) == [first, latest]
    calls = []
    monkeypatch.setattr(daily, "main", lambda options: calls.append(options) or 0)
    assert consumer.run(config, builder=daily, max_days=1) == 0
    assert len(calls) == 1
    calls.clear()
    assert consumer.run(config, builder=daily, max_days=3) == 0
    assert [options[options.index("--date") + 1] for options in calls] == [str(first), str(latest)]


def test_consumer_refuses_existing_product_without_marker(daily_input, tmp_path, monkeypatch):
    config = _config(daily_input, tmp_path)
    _, output, marker = consumer.product_paths(config, daily_input[1], builder=daily)
    output.mkdir(parents=True)
    monkeypatch.setattr(daily, "main", lambda _: pytest.fail("must not overwrite existing product"))
    assert consumer.run(config, builder=daily, write=True) == 1
    assert output.is_dir() and not marker.exists()
