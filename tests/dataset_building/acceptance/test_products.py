"""YAML standards checked against independent server and production baselines."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import MISSING, fields, replace
from pathlib import Path

import numpy as np
import pytest

from datakit.schema.datasets import specifications
from datakit.schema.datasets.specifications import (
    ChannelDefinition,
    ChannelMetadataDefinition,
    DatasetProduct,
    DatasetSchema,
    GridDefinition,
    MaskDefinition,
)
from pipelines.dataset_building.support.inputs import load_validation_schema

from ._helpers import SCHEMA_DIRECTORY


@pytest.fixture(scope="module")
def schemas():
    return {
        path.stem: load_validation_schema(path) for path in sorted(SCHEMA_DIRECTORY.glob("*.yaml"))
    }


def test_ten_schemas_are_complete_structural_specs(schemas):
    expected_counts = {
        "cra_c71_p25_h6": 69448, "cra_c87_p25_h24": 16802, "cra_c87_1p5_h24": 17167,
        "era5_c87_p25_h24": 24472, "era5_c78_p25_h6": 67208, "era5_c116_p25_h6": 2308,
        "era5_land_c8_1p5_h24": 27394, "oras5_c88_1p5_h24": 11688,
        "oras5_c88_p25_h24": 11688, "s2s_c76_1p5_h24": 27866,
    }
    assert set(schemas) == set(expected_counts)
    assert sum(len(spec.product.channels) for spec, _ in schemas.values()) == 786
    for name, (spec, document) in schemas.items():
        assert spec.expected_shape[0] == expected_counts[name]
        assert spec.config_id == document["dataset_id"]
        assert spec.dataset_id == document["expected_root_dataset_id"]
        assert spec.template_version == "v1" and spec.schema_status == "approved"
        assert spec.validation_scope == "confirmed-format"
        assert not spec.quality and not spec.valid_regions
        assert not spec.conversion_required
        assert spec.production_evidence is None
        assert not spec.product.mask.require_provenance
        assert not spec.product.mask.check_external_masks
        assert spec.mask_derivation is None
        assert spec.schema_path.is_file() and len(spec.schema_sha256) == 64
        for auxiliary in spec.variant.auxiliaries:
            if auxiliary.format == "npy":
                assert spec.auxiliary_provenance[auxiliary.filename]


def test_all_product_orders_match_independently_approved_server_baseline(schemas):
    root = Path(__file__).resolve().parents[3]
    records = json.loads((root / "docs/validation/PRODUCT_BASELINE_2026-10-13.json").read_text())
    observations = {record["dataset_id"]: record for record in records}
    for spec, _ in schemas.values():
        observed = observations[spec.dataset_id]
        assert spec.product.channel_names == tuple(observed["channels"])
        lat, _ = spec.variant.grid.coordinates()
        assert (float(lat[0]), float(lat[-1])) == (observed["lat_first"], observed["lat_last"])
        assert bool((np.diff(lat) > 0).all()) == (spec.product.product_id == "cra.c71")


def test_era5_c116_production_definitions_are_preserved(schemas):
    from pipelines.dataset import _raw_truth
    from pipelines.dataset_building.production import era5_c116

    spec = schemas["era5_c116_p25_h6"][0]
    assert not hasattr(era5_c116, "PRODUCT")
    assert era5_c116.CHANNEL_COUNT == 116
    assert era5_c116.DYNAMIC_CHANNELS is era5_c116.CHANNELS
    assert era5_c116.CHANNELS == spec.product.channel_names
    assert _raw_truth._resolve_channels(None) == era5_c116.CHANNELS
    assert np.array_equal(era5_c116.TARGET_LAT, np.linspace(90, -90, 721, dtype="float32"))
    assert np.array_equal(era5_c116.TARGET_LON, np.arange(0, 360, 0.25, dtype="float32"))
    assert era5_c116.SOURCE_VARIABLES["u10m"] == ("u10m", "u10")
    assert era5_c116.SOURCE_DIRECTORIES["swvl1"] == "vsw1"
    assert _raw_truth._source_spec("t850") == ("pl", "t", "t", 850)
    assert _raw_truth._source_spec("w10") == ("pl", "w", "w", 10)
    assert _raw_truth._source_spec("swvl1") == ("soil", "vsw1", "swvl1", None)
    assert _raw_truth._source_spec("wmb") == ("wave", "wmb", "wmb", None)


def test_yaml_definitions_initialize_in_a_fresh_process():
    code = """
from pathlib import Path
import sys
from pipelines.dataset_building.support.inputs import load_validation_schema
spec, _ = load_validation_schema(Path('datakit/schema/datasets/era5_c116_p25_h6.yaml'))
assert len(spec.product.channels) == 116
assert 'pipelines.dataset_building.production.era5_c116' not in sys.modules
assert 'pipelines.dataset_building.production.channel_metadata' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0, result.stderr


def test_oras5_standards_match_only_observed_server_datasets(schemas):
    for name, shape in (("oras5_c88_1p5_h24", (121, 240)), ("oras5_c88_p25_h24", (721, 1440))):
        spec = schemas[name][0]
        assert spec.product.version == "1.0" and spec.variant.cadence_hours == 24
        assert spec.variant.grid.shape == shape
        definitions = {channel.name: channel for channel in spec.product.channels}
        for variable in ("os", "ot", "ou", "ov"):
            channel = definitions[f"{variable}19978"]
            assert channel.variable == channel.source_name == variable
            assert channel.level == 199.78 and channel.level_units == "m"
            assert channel.level_type == "depth_below_sea_surface"
        assert definitions["os19978"].units is None
        assert definitions["os19978"].unconfirmed_fields == ("units", "source_units")
        assert "so19979" not in definitions


def test_production_metadata_is_not_the_acceptance_source():
    from pipelines.dataset_building.production.channel_metadata import build_channel_info

    names = ("so19979", "to50", "uo50", "vo50", "stl3", "swvl4", "q2m")
    info = build_channel_info(names, mean=np.zeros(7, dtype="f4"), std=np.ones(7, dtype="f4"))
    assert info["so19979"]["source_name"] == "so_oras"
    assert info["so19979"]["level"] == 199.79
    assert info["stl3"]["level_type"] == "soil_layer" and info["stl3"]["level"] == 3
    assert info["swvl4"]["units"] == "m3 m-3"
    assert info["q2m"]["units"] == "g kg-1"
    with pytest.raises(ValueError):
        build_channel_info(names, mean=np.zeros(7, dtype="f4"), std=np.full(7, np.inf, dtype="f4"))


def test_each_schema_declares_the_embedded_mask(schemas):
    for spec, _ in schemas.values():
        mask = spec.product.mask
        assert mask.channels == ("land_mask", "sea_mask")
        assert mask.dtype == "uint8" and mask.allowed_values == (0, 1)
        assert mask.shape(spec.variant.grid) == (2, *spec.variant.grid.shape)


def test_invalid_mask_schema_fails_before_dataset_io(schemas):
    mask = schemas["era5_land_c8_1p5_h24"][0].product.mask
    for changes in (
        {"channels": ("land_mask", "land_mask")}, {"dtype": "float32"},
        {"allowed_values": (0, 1), "complement_sum": 2},
        {"dimensions": ("mask_channel", "lon", "lat")}, {"array_name": "data"},
        {"require_provenance": "false"},
    ):
        with pytest.raises(ValueError):
            replace(mask, **changes)


@pytest.mark.parametrize(
    ("model", "required"),
    [
        (GridDefinition, ("north", "west", "latitude_direction")),
        (
            MaskDefinition,
            (
                "dimensions", "channel_dimensions", "dtype", "channel_dtype",
                "channels", "allowed_values", "valid_value", "complement_sum",
            ),
        ),
        (
            ChannelMetadataDefinition,
            (
                "attribute", "version_attribute", "version", "require_exact_channels",
                "long_name_required", "scale_offset_formula",
            ),
        ),
        (ChannelDefinition, ("variable_type", "preprocessing")),
        (DatasetProduct, ("mask",)),
        (
            DatasetSchema,
            (
                "template_version", "schema_version", "content_version", "zarr_format",
                "validation_scope", "schema_status", "arrays", "data_attributes",
                "channel_metadata", "unregistered_auxiliary_status",
            ),
        ),
    ],
)
def test_models_require_explicit_yaml_declarations(model, required):
    declared = {item.name: item for item in fields(model)}
    for name in required:
        assert declared[name].default is MISSING
        assert declared[name].default_factory is MISSING


@pytest.mark.parametrize("name", ["standard_arrays", "DATA_ATTRIBUTES", "STANDARD_LAND_SEA_MASK"])
def test_python_default_standard_copies_are_removed(name):
    from datakit.schema import datasets

    assert not hasattr(specifications, name)
    assert not hasattr(datasets, name)
    assert name not in datasets.__all__
