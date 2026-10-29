"""Protect the dataset-building workflow's dependency and configuration boundaries."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from datakit.schema.datasets.inputs import resolve_config_path
from pipelines.dataset_building.support.inputs import load_validation_schema

ROOT = Path(__file__).resolve().parents[3]


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


def test_dataset_schema_package_is_usable_without_workflow_dependencies():
    code = """
import importlib.abc
import sys
from importlib.resources import files
from pathlib import Path

class RejectWorkflowImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pipelines', 'examples', 'torch', 'xmetai'}:
            raise AssertionError('Unexpected schema dependency: ' + fullname)
        return None

sys.meta_path.insert(0, RejectWorkflowImports())
from datakit.schema.datasets import DatasetSchema, load_schema, verify_schema_unchanged
from datakit.schema.datasets import specifications, inputs
assert DatasetSchema.__module__ == specifications.__name__
assert load_schema.__module__ == inputs.__name__
assert not any(path.name.endswith(('.yaml', '.yml'))
               for path in files('datakit.schema.datasets').iterdir())
resources = sorted(Path(sys.argv[1]).glob('*.yaml'))
assert len(resources) == 10
channels = 0
for path in resources:
    schema, locations = load_schema(path)
    assert type(schema) is DatasetSchema
    assert not hasattr(schema, 'production_evidence')
    assert not hasattr(schema, 'snapshot_id')
    assert not hasattr(schema, 'execution_config_path')
    assert locations['dataset_id'] == schema.config_id
    verify_schema_unchanged(schema)
    channels += len(schema.product.channels)
assert channels == 786
assert not any(name == 'pipelines' or name.startswith('pipelines.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "examples/dataset_building/configs/standards")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_schema_and_runtime_definitions_have_distinct_single_homes():
    from datakit.schema.datasets import DatasetSchema, load_schema
    from datakit.schema.datasets import inputs as schema_inputs
    from pipelines.dataset_building.support import inputs, specifications

    assert specifications.DatasetValidationSpec.__bases__ == (DatasetSchema,)
    assert inputs.load_schema is load_schema is schema_inputs.load_schema
    assert inputs.resolve_config_path is schema_inputs.resolve_config_path
    schema_directory = ROOT / "datakit" / "schema" / "datasets"
    assert {path.name for path in schema_directory.glob("*.py")} == {
        "__init__.py", "specifications.py", "inputs.py",
    }
    for path in schema_directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = (
                [node.module or ""] if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                name.split(".")[0] in {"pipelines", "examples", "xarray", "zarr"}
                for name in modules
            ), (path, node.lineno)
    for module, expected in (
        (specifications, {"DatasetValidationSpec", "ConversionChannelRule", "ConversionValidationSpec"}),
        (inputs, {
            "_reject_retired_model_fields", "read_frozen_json", "load_validation_schema",
            "apply_execution", "load_schema_reference", "load_validation_run",
            "verify_execution_unchanged", "validation_input_files", "load_conversion_run",
        }),
    ):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        assert {
            node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        } == expected
    for name in ("DatasetValidationSpec", "ConversionChannelRule", "ConversionValidationSpec"):
        assert not hasattr(sys.modules[DatasetSchema.__module__], name)
    for name in ("load_validation_schema", "load_validation_run", "load_conversion_run"):
        assert not hasattr(schema_inputs, name)


def test_established_pipeline_entry_points_forward_to_dataset_building():
    from pipelines import dataset, dataset_building

    for name in (
        "validate_standard_dataset",
        "validate_standard_conversion",
    ):
        assert getattr(dataset, name) is getattr(dataset_building, name)


def test_specialist_configs_reference_loadable_schemas():
    directory = ROOT / "examples/dataset_building/configs"
    checked = 0
    for path in directory.glob("*.json"):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "target_schema" in config:
            schema = resolve_config_path(path, config["target_schema"])
            assert schema.is_file(), (path, schema)
            load_validation_schema(schema)
            checked += 1
    assert checked == 1


def test_workflow_modules_do_not_import_model_dependencies():
    code = """
import sys
from pipelines.dataset_building.workflows import dataset, conversion
from pipelines.dataset_building.support import reports
from datakit.schema.datasets import DatasetSchema
from pipelines.dataset_building.support.specifications import DatasetValidationSpec
assert not hasattr(DatasetSchema, 'production_evidence')
assert not hasattr(DatasetSchema, 'consumer_evidence')
assert not hasattr(DatasetSchema, 'consumer_required')
assert not hasattr(DatasetSchema, 'conversion_evidence')
assert hasattr(DatasetValidationSpec, 'production_evidence')
assert not hasattr(DatasetValidationSpec, 'consumer_evidence')
assert not any(name.startswith('pipelines.dataset_building.production') for name in sys.modules)
assert 'torch' not in sys.modules
assert 'xmetai' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_existing_data_entry_points_work_without_retired_model_modules():
    code = """
import importlib
import importlib.abc
import sys

class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'xmetai'} or fullname.endswith('.consumer'):
            raise AssertionError('Unexpected model dependency: ' + fullname)
        return None

sys.meta_path.insert(0, RejectModelImports())
from pipelines import dataset, dataset_building
assert not hasattr(dataset_building, 'check_core_consumer')
assert len(dataset_building.__all__) == 4
for name in ('validate_standard_dataset', 'validate_standard_conversion'):
    assert callable(getattr(dataset, name))
    assert getattr(dataset, name) is getattr(dataset_building, name)
for name in ('era5_nc_to_zarr', 'oras5_nc_to_zarr_88'):
    importlib.import_module('examples.dataset_building.' + name)
truth = importlib.import_module('examples.dataset_processing.era5_extract_raw_truth_day')
assert len(truth.subset_channels('core-70')) == 70
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for path in (
        "pipelines/dataset_building/acceptance/consumer.py",
        "examples/dataset_building/check_core_consumer.py",
        "examples/dataset_building/configs/core_land.json",
    ):
        assert not (ROOT / path).exists()


def test_combination_workflow_is_not_exposed_or_imported():
    code = """
import importlib.util
import sys
from pipelines import dataset, dataset_building
from pipelines.dataset_building.workflows import dataset as dataset_workflow, conversion
from pipelines.dataset_building.checks import format as checks
for module in (dataset, dataset_building, dataset_workflow, conversion):
    assert not hasattr(module, 'validate_standard_combination')
    assert not hasattr(module, 'run_combination_validation')
assert not hasattr(checks, 'check_standard_time_alignment')
assert importlib.util.find_spec('pipelines.dataset_building.workflows.combination') is None
assert not any(name.endswith('.combination') for name in sys.modules)
assert set(dataset_building.__all__) == {
    'validate_standard_dataset', 'validate_standard_conversion',
    'run_dataset_validation', 'run_conversion_validation',
}
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    directory = ROOT / "examples" / "dataset_building"
    assert not (directory / "validate_standard_combination.py").exists()
    assert not (directory / "configs" / "s2s_land_combination.json").exists()


def test_production_definitions_do_not_load_acceptance():
    code = """
import sys
from pipelines.dataset_building.production import era5_c116, channel_metadata
assert not any(
    name.startswith('pipelines.dataset_building.' + part)
    for part in ('checks', 'workflows', 'support') for name in sys.modules
)
assert 'torch' not in sys.modules
assert 'xmetai' not in sys.modules
assert not hasattr(era5_c116, 'PRODUCT')
assert era5_c116.CHANNEL_COUNT == 116
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_acceptance_implementation_has_one_workflow_local_home():
    assert not any((ROOT / "datakit").rglob("standard_dataset.py"))
    assert not (ROOT / "datakit" / "schema" / "loading.py").exists()
    for name in (
        "_standard_auxiliary.py", "_standard_quality.py", "standard_comparison.py",
        "evidence.py", "reports.py",
    ):
        assert not (ROOT / "datakit" / "validation" / name).exists()
    directory = ROOT / "pipelines" / "dataset_building"
    assert not list((directory / "definitions").glob("*.py"))
    assert not list((directory / "validation").glob("*.py"))
    assert sorted(path.name for path in directory.glob("*.py")) == ["__init__.py"]
    assert not list((ROOT / "datakit" / "schema" / "datasets").glob("*.yaml"))
    assert len(list((ROOT / "examples/dataset_building/configs/standards").glob("*.yaml"))) == 10


def test_acceptance_subpackages_have_one_implementation_per_role():
    directory = ROOT / "pipelines" / "dataset_building"
    expected = {
        "": {"__init__.py"},
        "production": {"__init__.py", "era5_c116.py", "channel_metadata.py"},
        "checks": {"__init__.py", "format.py", "values.py", "storage.py", "conversion.py"},
        "workflows": {"__init__.py", "dataset.py", "conversion.py"},
        "support": {"__init__.py", "specifications.py", "inputs.py", "reports.py"},
    }
    for name, filenames in expected.items():
        assert {path.name for path in (directory / name).glob("*.py")} == filenames
        if name:
            tree = ast.parse((directory / name / "__init__.py").read_text(encoding="utf-8"))
            assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    code = """
import importlib
from pipelines.dataset_building import validate_standard_dataset
from pipelines.dataset_building.workflows import dataset as dataset_validation
from pipelines.dataset_building.support import reports, specifications, inputs
from pipelines.dataset_building.checks import format, values, storage, conversion
for module, names in (
    (reports, ('ValidationReport', 'write_reports', 'ensure_report_destination',
               'report_exit_code')),
    (inputs, ('load_validation_schema', 'load_validation_run', 'load_conversion_run')),
    (specifications, ('DatasetValidationSpec', 'ConversionChannelRule', 'ConversionValidationSpec')),
    (format, ('check_dataset_format', 'check_auxiliary_format', 'check_channel_semantics')),
    (values, ('StandardQualityAccumulator', 'check_auxiliary_values', 'check_statistics')),
    (storage, ('check_root_metadata', 'check_array_encoding', 'check_production_receipt')),
    (conversion, ('compare_channel_values', 'attach_conversion_evidence')),
    (dataset_validation, ('scan_standard_blocks', '_resident_memory_bytes',
               '_peak_memory_bytes', '_fingerprint')),
):
    assert all(callable(getattr(module, name)) for name in names)
assert validate_standard_dataset is dataset_validation.validate_standard_dataset
assert validate_standard_dataset.__module__ == dataset_validation.__name__
assert not hasattr(reports, 'attach_conversion_evidence')
for name in ('acceptance', 'definitions', 'validation', 'runners', 'dataset_validation'):
    assert importlib.util.find_spec('pipelines.dataset_building.' + name) is None
assert importlib.util.find_spec('pipelines.dataset_building.production.channel_attributes') is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (directory / "acceptance").exists()


def test_validation_examples_only_adapt_cli_arguments():
    directory = ROOT / "examples" / "dataset_building"
    for name in (
        "validate_standard_dataset.py", "validate_standard_conversion.py",
    ):
        tree = ast.parse((directory / name).read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or ""
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert not any(
            module.split(".")[0] in {"numpy", "xarray", "zarr", "torch", "xmetai"}
            for module in imports
        ), name


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("support", {"checks", "workflows", "production"}),
        ("checks", {"workflows", "production"}),
        ("production", {"checks", "workflows", "support"}),
        ("workflows", {"production"}),
    ],
)
def test_dataset_building_dependencies_follow_layer_direction(package, forbidden):
    from importlib.util import resolve_name

    directory = ROOT / "pipelines" / "dataset_building" / package
    prefix = "pipelines.dataset_building."
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = resolve_name(
                    "." * node.level + (node.module or ""), prefix + package,
                ) if node.level else node.module or ""
                imported = [module] + [module + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            assert not any(
                name == prefix + part or name.startswith(prefix + part + ".")
                for name in imported for part in forbidden
            ), (path, node.lineno, imported)


def test_check_rules_do_not_open_files_or_walk_directories():
    forbidden = {
        "open", "open_zarr", "open_group", "open_array", "open_dataset",
        "read_bytes", "read_text", "iterdir", "rglob", "glob", "stat", "is_file",
        "write_text", "write_bytes", "save", "load",
    }
    directory = ROOT / "pipelines" / "dataset_building" / "checks"
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name) else ""
            )
            assert name not in forbidden, (path, node.lineno, name)
