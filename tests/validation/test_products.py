"""Regression checks for standard dataset products."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from datakit.schema._standard_products import define_channels
from datakit.schema.dataset_products import get_dataset_product
from datakit.schema.standard_dataset import STANDARD_LAND_SEA_MASK


@pytest.mark.parametrize(
    "module_name,product_id",
    [
        ("cra_c71", "cra.c71"),
        ("cra_c87", "cra.c87"),
        ("era5_c78", "era5.c78"),
        ("era5_c87", "era5.c87"),
        ("era5_c116", "era5.c116"),
        ("era5_land_c8", "era5-land.c8"),
        ("oras5_c88", "oras5.c88"),
        ("s2s_c76", "s2s.c76"),
    ],
)
def test_product_modules_expose_the_same_public_declarations(module_name, product_id):
    from datakit.schema import DatasetProduct, DatasetVariant, MaskDefinition

    module = importlib.import_module(f"datakit.schema.{module_name}")
    assert isinstance(module.CHANNELS, tuple)
    assert all(isinstance(name, str) for name in module.CHANNELS)
    assert isinstance(module.MASK, MaskDefinition)
    assert isinstance(module.VARIANTS, tuple)
    assert all(isinstance(variant, DatasetVariant) for variant in module.VARIANTS)
    assert isinstance(module.PRODUCT, DatasetProduct)
    assert module.PRODUCT is get_dataset_product(product_id)
    assert module.PRODUCT.channel_names == module.CHANNELS
    assert module.PRODUCT.mask is module.MASK
    assert module.PRODUCT.variants is module.VARIANTS


def test_era5_c116_legacy_production_exports_remain_compatible():
    from datakit.schema import DYNAMIC_CHANNELS, TARGET_LAT, TARGET_LON, era5_c116
    from datakit.schema.dataset_products import ERA5_C116

    assert ERA5_C116 is era5_c116.PRODUCT
    assert DYNAMIC_CHANNELS is era5_c116.CHANNELS
    assert era5_c116.CHANNEL_COUNT == 116
    assert era5_c116.BASE_LEVELS == (
        1000,
        925,
        850,
        700,
        600,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        50,
    )
    assert TARGET_LAT.dtype == TARGET_LON.dtype == np.dtype("float32")
    assert np.array_equal(TARGET_LAT, np.linspace(90, -90, 721, dtype="float32"))
    assert np.array_equal(TARGET_LON, np.arange(0, 360, 0.25, dtype="float32"))
    assert era5_c116.SOURCE_VARIABLES["u10m"] == ("u10m", "u10")
    assert era5_c116.SOURCE_DIRECTORIES["swvl1"] == "vsw1"
    assert era5_c116.GROUPS["swvl1"] == "soil"


def test_schema_imports_initialize_in_a_fresh_process():
    import subprocess
    import sys

    code = (
        "from datakit.schema._standard_products import define_channels; "
        "from datakit.schema.era5_c116 import PRODUCT; "
        "from datakit.schema import get_dataset_product; "
        "assert PRODUCT is get_dataset_product('era5.c116'); "
        "assert len(define_channels(('t2m',))) == 1"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr


def test_product_semantics_are_explicit_and_versioned():
    assert get_dataset_product("era5-land.c8").channel_names == (
        "stl1",
        "stl2",
        "stl3",
        "stl4",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
    )
    for identifier, count in (
        ("cra.c71", 71),
        ("cra.c87", 87),
        ("era5.c78", 78),
        ("era5.c87", 87),
        ("era5.c116", 116),
        ("era5-land.c8", 8),
        ("oras5.c88", 88),
        ("s2s.c76", 76),
    ):
        product = get_dataset_product(identifier)
        assert len(product.channel_names) == count and len(set(product.channel_names)) == count
    definitions = {
        d.name: d
        for d in define_channels(
            ("swvl1", "stl4", "q2m", "u10m", "t2m_min", "q1000", "os19978", "so19979")
        )
    }
    assert definitions["swvl1"].level_type == "soil_layer"
    assert definitions["q2m"].units == "g kg-1"
    assert definitions["t2m_min"].level_type == "height_above_ground"
    assert definitions["os19978"].level == 199.78
    assert "so19979" in get_dataset_product("oras5.c88", "2.0").channel_names
    assert "so19979" not in get_dataset_product("oras5.c88", "1.0").channel_names
    for name in ("unknown1", "q2", "os19979", "so19978"):
        with pytest.raises(ValueError):
            define_channels((name,))


def test_ten_profiles_are_complete_structural_specs():
    from examples.validation._standard_config import load_profile

    directory = Path(__file__).resolve().parents[2] / "examples/validation/profiles"
    inventory = json.loads((directory / "inventory.json").read_text())
    assert len(inventory["profiles"]) == 10
    expected_counts = [69448, 16802, 17167, 24472, 67208, 2308, 27394, 11688, 11688, 27866]
    for filename, count in zip(inventory["profiles"], expected_counts, strict=True):
        spec, document = load_profile(directory / filename)
        assert spec.expected_shape[0] == count
        assert document["profile_status"] == "approved"
        assert spec.validation_scope == "confirmed-format"
        assert not spec.quality and not spec.valid_regions
        assert not spec.consumer_required and not spec.conversion_required
        assert spec.production_evidence is None
        assert (
            not spec.product.mask.require_provenance and not spec.product.mask.check_external_masks
        )
        assert spec.mask_derivation is None


def test_known_soil_metadata_and_std_infinity_are_not_accepted_as_truth():
    from datakit.schema import build_channel_info

    names = ("stl3", "swvl4", "q2m", "u10m", "t2m_min")
    info = build_channel_info(names, mean=np.zeros(5, dtype="f4"), std=np.ones(5, dtype="f4"))
    assert info["stl3"]["level_type"] == "soil_layer" and info["stl3"]["level"] == 3
    assert info["swvl4"]["units"] == "m3 m-3" and info["swvl4"]["source_name"] == "swvl4"
    assert info["q2m"]["units"] == "g kg-1" and info["t2m_min"]["level"] == 2
    with pytest.raises(ValueError):
        build_channel_info(names, mean=np.zeros(5, dtype="f4"), std=np.full(5, np.inf, dtype="f4"))


def test_all_product_orders_match_independently_approved_server_baseline():
    from examples.validation._standard_config import load_profile

    root = Path(__file__).resolve().parents[2]
    records = json.loads((root / "docs/validation/PRODUCT_BASELINE_2026-10-13.json").read_text())
    observations = {record["dataset_id"]: record for record in records}
    inventory = root / "examples/validation/profiles/inventory.json"
    for filename in json.loads(inventory.read_text())["profiles"]:
        spec, _ = load_profile(inventory.parent / filename)
        observed = observations[spec.dataset_id]
        assert spec.product.channel_names == tuple(observed["channels"])
        lat, _ = spec.variant.grid.coordinates()
        assert (float(lat[0]), float(lat[-1])) == (observed["lat_first"], observed["lat_last"])
        assert bool((np.diff(lat) > 0).all()) == (spec.product.product_id == "cra.c71")


def test_each_product_declares_the_same_mask_contract_with_its_own_grid():
    from datakit.schema.dataset_products import STANDARD_PRODUCTS

    for product in STANDARD_PRODUCTS:
        assert product.mask is STANDARD_LAND_SEA_MASK
        assert product.mask.channels == ("land_mask", "sea_mask")
        assert product.mask.dtype == "uint8" and product.mask.allowed_values == (0, 1)
        for variant in product.variants:
            assert product.mask.shape(variant.grid) == (2, *variant.grid.shape)
            layouts = product.mask.layouts(variant.grid)
            assert layouts["mask"] == (
                ("mask_channel", "lat", "lon"),
                (2, *variant.grid.shape),
                "uint8",
            )
            assert layouts["mask_channel"] == (("mask_channel",), (2,), "string")
    assert get_dataset_product("cra.c71").variant("p25.h6").grid.coordinates()[0][0] == -90
    assert get_dataset_product("era5.c116").variant("p25.h6").grid.coordinates()[0][0] == 90


def test_invalid_mask_schema_fails_before_dataset_io():
    for changes in (
        {"channels": ("land_mask", "land_mask")},
        {"dtype": "float32"},
        {"allowed_values": (0, 1), "complement_sum": 2},
        {"dimensions": ("mask_channel", "lon", "lat")},
        {"array_name": "data"},
        {"require_provenance": "false"},
    ):
        with pytest.raises(ValueError):
            replace(STANDARD_LAND_SEA_MASK, **changes)
