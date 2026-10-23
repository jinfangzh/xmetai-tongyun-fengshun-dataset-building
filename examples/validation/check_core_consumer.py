"""Exercise actual core readers/windows/batches in the installed core environment.

Model-specific forward/export tests provide an independently frozen receipt;
they cannot be inferred from generic Zarr metadata. No core imports enter the
datakit library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from datakit.validation.zarr_inspection import consolidated_mismatches, file_sha256, read_zarr_metadata
from datakit.validation.evidence import CONSUMER_CHECKS
from datakit.validation import to_jsonable
from ._standard_config import load_profile, read_json


def run_consumer(config: dict, config_sha256: str) -> dict:
    import torch
    import xmetai
    from xmetai.data.weather.data_util import get_data_info, make_seq
    from xmetai.data.weather.multi_zarr_dataset import MultiZarrDataset

    spec, document = load_profile(Path(config["profile"]))
    path = Path(config.get("dataset_path") or document["dataset_path"])
    core_root = Path(xmetai.__file__).resolve().parent.parent
    commit = subprocess.run(
        ["git", "-C", str(core_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    receipt = {
        "dataset_id": spec.dataset_id,
        "data_revision": spec.data_revision,
        "snapshot_id": spec.snapshot_id,
        "product_id": spec.product.product_id,
        "product_version": spec.product.version,
        "profile_version": spec.profile_version,
        "consumer_version": commit or getattr(xmetai, "__version__", "unidentified"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "zarr": zarr.__version__,
        },
        "config_sha256": config_sha256,
        "mapping": config,
        "checks": {
            name: {"status": "not-run", "evidence": "Not exercised in generic reader probe"}
            for name in CONSUMER_CHECKS
        },
    }
    checks = receipt["checks"]

    def result(name, passed, observed=None, evidence="Actual installed core execution"):
        checks[name] = {
            "status": "pass" if passed else "fail",
            "observed": observed,
            "evidence": evidence,
        }

    result("environment", True, receipt["environment"])
    metadata = read_zarr_metadata(path)
    mismatches = consolidated_mismatches(metadata)
    result(
        "metadata",
        mismatches == [],
        mismatches,
        "Fixed consolidated=True reader requires valid consolidated metadata",
    )
    options = dict(config["dataset"])
    if "data_paths" in options:
        raise ValueError(
            "This probe accepts one path; test joint consumers with an explicit combination configuration"
        )
    options.setdefault("return_valid_mask", True)
    options.setdefault("training", False)
    frames = (
        int(options["hist_frames"])
        + int(max(np.atleast_1d(options["fcst_frames"])))
        + int(options.get("start_frames", 0))
    )
    if options.get("interval", 1) < 1:
        raise ValueError("Reader probe requires regular positive interval windows")
    cadence = spec.variant.cadence_hours
    times = pd.DatetimeIndex(spec.expected_times())
    expected_starts = np.arange(
        0, len(times) - frames + 1, int(options.get("interval", 1)), dtype="int32"
    )
    actual_starts = make_seq(times, frames, freq=cadence, interval=int(options.get("interval", 1)))
    exact = make_seq(times[:frames], frames, freq=cadence)
    broken = times[: frames + 2].to_numpy().copy()
    broken[1] += np.timedelta64(1, "h")
    broken_starts = make_seq(pd.DatetimeIndex(broken), frames, freq=cadence)
    window_ok = (
        options.get("freq", cadence) == cadence
        and options.get("freq_unit", "hours") == "hours"
        and np.array_equal(actual_starts, expected_starts)
        and np.array_equal(exact, np.array([0], dtype="i4"))
        and 0 not in broken_starts
    )
    result(
        "time_windows",
        window_ok,
        {
            "expected_count": len(expected_starts),
            "actual_count": len(actual_starts),
            "exact_F": exact,
            "broken_interval_starts": broken_starts,
            "expected_last": int(expected_starts[-1]) if len(expected_starts) else None,
            "actual_last": int(actual_starts[-1]) if len(actual_starts) else None,
        },
        "Independent N-F+1 coverage, exactly-F input, final start and gap rejection",
    )
    info = get_data_info([str(path)])
    with xr.open_zarr(str(path), consolidated=False, chunks=None, mask_and_scale=False) as raw:
        names = tuple(info["in_names"])
        buffers = info["buffers"]
        stats_ok = names == spec.product.channel_names and all(
            key in buffers
            and buffers[key].shape == (len(names), 1, 1)
            and np.array_equal(buffers[key][:, 0, 0], raw[key].values)
            for key in ("mean", "std")
        )
        masks_ok = all(
            key in buffers and np.array_equal(buffers[key], raw["mask"].values[index])
            for key, index in (("land_mask", 0), ("mask", 1))
        )
        result(
            "channel_layout", stats_ok and raw.data.dims == ("time", "channel", "lat", "lon"), names
        )
        spatial_ok = masks_ok and not options.get("lat_flip", False) and options.get("bbox") is None
        result(
            "spatial_alignment",
            spatial_ok,
            evidence="Actual buffers aligned to raw grid; transformed bbox/flip requires separate model buffer synchronization tests",
        )
        expected_land = config.get("expected_land_names")
        expected_sea = config.get("expected_sea_names")
        if expected_land is not None and expected_sea is not None:
            result(
                "channel_semantics",
                [names[i] for i in info["landid"]] == expected_land
                and [names[i] for i in info["maskid"]] == expected_sea,
                {
                    "actual_land": [names[i] for i in info["landid"]],
                    "actual_sea": [names[i] for i in info["maskid"]],
                },
            )
        dataset = None
        try:
            estimated_bytes = (
                frames * len(names) * np.prod(spec.variant.grid.shape) * 12 + 512 * 1024**2
            )
            if estimated_bytes > config.get("memory_budget_mib", 4096) * 1024**2:
                raise ValueError(
                    "Consumer frame/batch estimate exceeds declared process memory budget"
                )
            dataset = MultiZarrDataset(data_paths=[str(path)], **options)
            sample = dataset[0]
            selected = options.get("in_names") or list(names)
            targets = options.get("target_names") or selected
            lat, lon = dataset.ds.lat.values, dataset.ds.lon.values
            input_shape = (int(options["hist_frames"]), len(selected), len(lat), len(lon))
            batch_ok = tuple(sample["inputs"].shape) == input_shape and tuple(dataset.ds.dims) == (
                "time",
                "channel",
                "lat",
                "lon",
            )
            if not options.get("inference_only", False):
                batch_ok = batch_ok and sample["targets"].shape[1:] == (
                    len(targets),
                    len(lat),
                    len(lon),
                )
            first_position = int(dataset.inds[0])

            def compare_frame(tensor, valid_tensor, position, channel_names):
                values = raw.data.sel(
                    time=dataset.ds.time.values[position], channel=channel_names, lat=lat, lon=lon
                ).values
                if options.get("lat_flip", False):
                    values = values[:, ::-1, :]
                correct = tensor.dtype == torch.float32 and tensor.shape == values.shape
                for ci, name in enumerate(channel_names):
                    channel = values[ci].astype("f4")
                    valid = np.isfinite(channel)
                    if name in options.get("zero_fill_invalid_names", []):
                        valid &= channel != 0
                    np.nan_to_num(channel, copy=False, nan=0, posinf=0, neginf=0)
                    correct = np.array_equal(tensor[ci].cpu().numpy(), channel) and correct
                    if valid_tensor is not None:
                        correct = np.array_equal(valid_tensor[ci].cpu().numpy(), valid) and correct
                return correct

            batch_ok = (
                compare_frame(
                    sample["inputs"][0],
                    sample.get("input_valid_mask", [None])[0],
                    first_position,
                    selected,
                )
                and batch_ok
            )
            if not options.get("inference_only", False):
                lead = int(np.atleast_1d(options["fcst_frames"])[0]) + int(
                    options.get("start_frames", 0)
                )
                target_position = first_position + int(options["hist_frames"]) + lead - 1
                batch_ok = (
                    compare_frame(
                        sample["targets"][0],
                        sample.get("target_valid_mask", [None])[0],
                        target_position,
                        targets,
                    )
                    and batch_ok
                )
            result(
                "batch",
                bool(batch_ok),
                {k: tuple(v.shape) for k, v in sample.items()},
                "Actual __getitem__(0), raw first input/target frame values and validity, dtype/channel/shape/lead mapping",
            )
        except Exception as exc:
            result("batch", False, {"type": type(exc).__name__, "message": str(exc)})
        finally:
            if dataset is not None:
                dataset.ds.close()
    checks["multi_dataset"] = {
        "status": "not-applicable",
        "evidence": "One dataset selected; joint use requires a separate consumer configuration",
    }
    model_reference = config.get("model_evidence")
    if model_reference:
        file = Path(model_reference["path"])
        if file_sha256(file) != model_reference["sha256"]:
            raise ValueError("Frozen model receipt hash mismatch")
        model = read_json(file)
        binding = (
            "dataset_id",
            "data_revision",
            "snapshot_id",
            "product_id",
            "product_version",
            "consumer_version",
        )
        if any(model.get(key) != receipt[key] for key in binding):
            raise ValueError("Model receipt identity/core version differs from reader probe")
        for name in ("model_parameters", "forward", "inverse", "output"):
            item = model.get("checks", {}).get(name)
            if isinstance(item, dict):
                checks[name] = item
    return to_jsonable(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Actual core consumer probe; model gates retain not-run until real model evidence is supplied"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    spec, document = load_profile(Path(config["profile"]))
    source = Path(config.get("dataset_path") or document["dataset_path"]).resolve()
    if args.output.resolve().is_relative_to(source):
        raise ValueError("Consumer report must be outside input data")
    receipt = run_consumer(config, hashlib.sha256(args.config.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"receipt={args.output} sha256={file_sha256(args.output)}")
    return (
        1
        if any(c["status"] == "fail" for c in receipt["checks"].values())
        else (
            2
            if any(
                c["status"] not in ("pass", "not-applicable") for c in receipt["checks"].values()
            )
            else 0
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
