"""Daily boundaries, isolated discovery, and normalized-product dry runs."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.schema import load_mapping, load_schema
from datakit.schema.semantics import TemporalKind
from examples.dataset_building import example_build_era5 as era5
from examples.dataset_building import example_build_tongyun_daily as daily
from pipelines.dataset import build_zarr


@pytest.fixture
def daily_input(tmp_path):
    description = io.get_description(daily.SOURCE_CONFIG)
    mapping = load_mapping(description)
    day = date(2025, 1, 1)  # Include the year boundary.
    root = tmp_path / "raw" / "daily"
    for variable, template in description["layout"]["files"].items():
        declaration = mapping.variables[variable]
        accumulated = declaration.temporal_kind == TemporalKind.INTERVAL_ACCUMULATION
        for selected_day in ([day - timedelta(days=1), day] if accumulated else [day]):
            start = np.datetime64(selected_day.isoformat(), "h")
            times = start + np.arange(24 if accumulated else 4) * np.timedelta64(1 if accumulated else 6, "h")
            coords = {"valid_time": times, "latitude": [90., 89.75], "longitude": [0., .25]}
            dims = ["valid_time"]
            if variable in {"z", "t", "u", "v", "q"}:
                coords["pressure_level"] = ("pressure_level", [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], {"units": "hPa"})
                dims.append("pressure_level")
            dims += ["latitude", "longitude"]
            shape = [len(coords[d][1]) if isinstance(coords[d], tuple) else len(coords[d]) for d in dims]
            field = xr.DataArray(np.ones(shape, dtype="f4"), dims=dims, coords=coords,
                                 name=declaration.source, attrs={"units": declaration.source_unit})
            path = root / template.format(day=datetime.combine(selected_day, time()))
            path.parent.mkdir(parents=True, exist_ok=True)
            field.to_dataset().to_netcdf(path, engine="scipy")
    return root, day, description


def test_daily_cli_stages_only_required_partitions(daily_input, tmp_path, monkeypatch):
    root, day, description = daily_input
    (root / "ignored.nc.part").touch()
    stages = []

    def check_build(args):
        stage = Path(args[args.index("--source") + 1])
        stages.append(stage)
        assert len(list(stage.rglob("*.nc"))) == 23
        assert all(p.is_symlink() for p in stage.rglob("*.nc"))
        assert len(io.open(stage, description).discover(all_files=True)) == 23
        assert "--dry-run" in args and "--write" not in args
        assert args[args.index("--start-time") + 1] == "2025-01-01T00:00"
        assert args[args.index("--end-time") + 1] == "2025-01-01T18:00"
        return 0

    monkeypatch.setattr(era5, "main", check_build)
    args = ["--date", str(day), "--source", str(root), "--destination", str(tmp_path / "result.zarr"),
            "--report-dir", str(tmp_path / "reports")]
    for name in ("mean-nc", "std-nc", "mask-nc", "const-nc", "weight-npy"):
        args.extend([f"--{name}", str(tmp_path / name)])
    assert daily.main(args) == 0
    assert not stages[0].exists()
    assert len(list(root.rglob("*.nc"))) == 23


def test_missing_previous_day_is_reported_before_build(daily_input):
    root, day, description = daily_input
    boundary = root / "sfc_accum_h1/total_precipitation/2024/total_precipitation_20241231.nc"
    boundary.unlink()
    with pytest.raises(FileNotFoundError, match="previous day's.*hourly"):
        daily.required_files(root, day, description)


def test_daily_partitions_build_normalized_c78_in_dry_run(daily_input, tmp_path):
    root, day, description = daily_input
    spec, _ = load_schema(daily.TARGET_SPEC)
    grid = replace(spec.variant.grid, latitude_count=2, longitude_count=2)
    product = replace(spec.product, variants=(replace(spec.variant, grid=grid, auxiliaries=()),))
    spec = replace(spec, product=product, first_time=f"{day}T00", last_time=f"{day}T18",
                   directory_name="result.zarr", identity_mapping_evidence="Synthetic daily test")
    channels = list(spec.product.channel_names)
    mean = xr.DataArray(np.zeros(78, dtype="f4"), dims="channel", coords={"channel": channels})
    lat, lon = grid.coordinates()
    mask = xr.DataArray(np.array([np.ones((2, 2)), np.zeros((2, 2))], dtype="u1"),
                        dims=("mask_channel", "lat", "lon"),
                        coords={"mask_channel": ["land_mask", "sea_mask"], "lat": lat, "lon": lon})
    paths = {}
    for name, array in {"mean": mean, "std": xr.ones_like(mean), "mask": mask}.items():
        paths[name] = tmp_path / f"{name}.nc"
        array.rename(name).to_dataset().to_netcdf(paths[name], engine="scipy")
    process = partial(era5.process, mean_path=paths["mean"], std_path=paths["std"], mask_path=paths["mask"])
    with era5.read_source(root, description, spec=spec) as raw:
        result = process(raw, description, spec)
        assert result.data.shape == (4, 78, 2, 2)
        np.testing.assert_allclose(result.data.sel(channel="tp").compute(), np.log1p(6000), rtol=1e-3)
        np.testing.assert_allclose(result.data.sel(channel="ssr").compute(), 6 / 21600, rtol=1e-3)
    output = tmp_path / "result.zarr"
    report = build_zarr(description, spec, root, output, read=era5.read_source, process=process, dry_run=True)
    assert report.coverage["conformance_passed"], report.to_dict()
    assert not output.exists()


def _ready_config(daily_input, tmp_path):
    import json
    root, day, description = daily_input
    paths = [original for original, _ in daily.required_files(root, day, description)]
    status = tmp_path / "status.jsonl"
    status.write_text("".join(json.dumps({"status": "completed", "target": str(p)}) + "\n"
                              for p in paths), encoding="utf-8")
    return {"source": root, "download_status": status, "output_root": tmp_path / "products",
            "report_dir": tmp_path / "reports",
            **{n: tmp_path / n for n in ("mean_nc", "std_nc", "mask_nc", "const_nc", "weight_npy")}}


def test_ready_consumer_waits_for_all_completed_records(daily_input, tmp_path):
    from examples.dataset_building import build_ready_tongyun_days as consumer
    config = _ready_config(daily_input, tmp_path)
    completed = consumer.completed_paths(config["download_status"])
    assert list(consumer.ready_days(config, completed)) == [daily_input[1]]
    completed.remove(next(iter(completed)))
    assert list(consumer.ready_days(config, completed)) == []
    # An incomplete concurrently written tail does not hide previous success records.
    with config["download_status"].open("a") as stream:
        stream.write('{"status":')
    assert len(consumer.completed_paths(config["download_status"])) == 23


def test_ready_consumer_publishes_once_after_success(daily_input, tmp_path, monkeypatch):
    from examples.dataset_building import build_ready_tongyun_days as consumer
    config = _ready_config(daily_input, tmp_path)
    calls = []

    def fake_build(options):
        calls.append(options)
        # Simulate a successful writer; no actual Zarr payload is written.
        Path(options[options.index("--destination") + 1]).mkdir(parents=True)
        return 0

    monkeypatch.setattr(daily, "main", fake_build)
    assert consumer.run(config, write=True) == 0
    assert len(list((config["output_root"] / "_completed").glob("*.json"))) == 1
    assert (config["output_root"] / "era5.tongyun.20250101.c78.p25.h6.v1").is_dir()
    assert consumer.run(config, write=True) == 0
    assert len(calls) == 1


def test_ready_consumer_retries_failed_attempt_without_overwrite(daily_input, tmp_path, monkeypatch):
    from examples.dataset_building import build_ready_tongyun_days as consumer
    config = _ready_config(daily_input, tmp_path)
    attempts = []

    def fake_build(options):
        output = Path(options[options.index("--destination") + 1])
        attempts.append(output)
        assert not output.exists()
        output.mkdir(parents=True)
        return 1 if len(attempts) == 1 else 0

    monkeypatch.setattr(daily, "main", fake_build)
    assert consumer.run(config, write=True) == 1
    assert not (config["output_root"] / "_completed").exists()
    assert consumer.run(config, write=True) == 0
    assert attempts[0] != attempts[1]
    assert attempts[0].is_dir()  # Preserve failed products for diagnosis.


def test_ready_consumer_dry_run_does_not_mark_done(daily_input, tmp_path, monkeypatch):
    from examples.dataset_building import build_ready_tongyun_days as consumer
    config = _ready_config(daily_input, tmp_path)
    calls = []
    monkeypatch.setattr(daily, "main", lambda options: calls.append(options) or 0)
    assert consumer.run(config) == 0
    assert "--dry-run" in calls[0]
    assert not config["output_root"].exists()
