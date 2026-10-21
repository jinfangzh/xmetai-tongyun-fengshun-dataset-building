"""Protect the dataset-building workflow's dependency and configuration boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from pipelines.dataset_building.config import load_profile


ROOT = Path(__file__).resolve().parents[2]


def test_base_library_does_not_load_dataset_building():
    code = """
import sys
import datakit
from datakit import io, schema, transforms, validation
from datakit.schema import dimensions, variables, semantics

assert not any(name == 'pipelines' or name.startswith('pipelines.') for name in sys.modules)
assert not hasattr(schema, 'DatasetProduct')
assert not hasattr(schema, 'get_dataset_product')
assert not hasattr(validation, 'ValidationReport')
assert not hasattr(schema, 'era5_c116')
assert not hasattr(schema, 'DYNAMIC_CHANNELS')
assert not hasattr(schema, 'build_channel_info')
assert 'datakit.schema.channel_attributes' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_established_pipeline_entry_points_forward_to_dataset_building():
    from pipelines import dataset, dataset_building

    for name in (
        "validate_standard_dataset",
        "validate_standard_conversion",
        "validate_standard_combination",
    ):
        assert getattr(dataset, name) is getattr(dataset_building, name)


def test_specialist_configs_reference_loadable_profiles():
    directory = ROOT / "examples/dataset_building/configs"
    checked = 0
    for path in directory.glob("*.json"):
        config = json.loads(path.read_text(encoding="utf-8"))
        for reference in [config, *config.get("members", [])]:
            for key in ("profile", "target_profile"):
                if key in reference:
                    profile = ROOT / reference[key]
                    assert profile.is_file(), (path, key, profile)
                    load_profile(profile)
                    checked += 1
    assert checked == 4
