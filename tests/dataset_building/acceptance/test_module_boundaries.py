"""Protect the dataset-building workflow's dependency and configuration boundaries."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

EXPECTED_PIPELINE_FILES = {
    "__init__.py",
    "build.py",
    "preparation.py",
    "validate.py",
    "_sources.py",
    "evidence.py",
    "append.py",
    "configuration.py",
    "create.py",
    "compare.py",
    "reports.py",
}


def test_base_library_does_not_load_dataset_building():
    code = """
import sys
import datakit
from datakit import io, schema, transforms, validation
from datakit.schema import dimensions, variables, semantics

assert not any(name == 'pipelines' or name.startswith('pipelines.') for name in sys.modules)
assert schema.DatasetProduct.__module__ == 'datakit.schema.specifications'
assert not hasattr(schema, 'get_dataset_product')
assert not hasattr(validation, 'ValidationReport')
assert not hasattr(schema, 'era5_c116')
assert not hasattr(schema, 'DYNAMIC_CHANNELS')
assert not hasattr(schema, 'build_channel_info')
assert 'datakit.schema.channel_attributes' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
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
from datakit.schema import DatasetSchema, load_schema, verify_schema_unchanged
from datakit.schema import specifications, parser
assert DatasetSchema.__module__ == specifications.__name__
assert load_schema.__module__ == parser.__name__
assert not any(path.name.endswith(('.yaml', '.yml'))
               for path in files('datakit.schema').iterdir())
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
        [sys.executable, "-c", code, str(ROOT / "config/dataset_specs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_schema_and_runtime_definitions_have_distinct_single_homes():
    from datakit.schema import DatasetSchema, load_schema
    from datakit.schema import parser as schema_parser
    from pipelines.dataset import configuration as inputs

    assert inputs.DatasetValidationSpec.__bases__ == (DatasetSchema,)
    assert inputs.load_schema is load_schema is schema_parser.load_schema
    assert inputs.resolve_config_path is schema_parser.resolve_config_path
    schema_directory = ROOT / "datakit" / "schema"
    # Git removes tracked source files, but may leave ignored __pycache__ on
    # other checkouts. Check duplicate implementations rather than directory existence.
    legacy_sources = sorted((schema_directory / "datasets").rglob("*.py"))
    assert not legacy_sources, f"旧 schema 目录仍有源码未迁移：{legacy_sources}"
    assert {path.name for path in schema_directory.glob("*.py")} == {
        "__init__.py",
        "specifications.py",
        "parser.py",
        "dimensions.py",
        "mapping.py",
        "semantics.py",
        "variables.py",
    }
    for path in (schema_directory / "specifications.py", schema_directory / "parser.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            forbidden = [
                name
                for name in modules
                if name.split(".")[0] in {"pipelines", "examples", "xarray", "zarr"}
            ]
            assert not forbidden, (
                f"Schema model/loading must not import {forbidden}: {path}:{node.lineno}"
            )
    for module, expected in (
        (
            inputs,
            {
                "DatasetValidationSpec",
                "_reject_retired_model_fields",
                "read_frozen_json",
                "load_validation_schema",
                "apply_execution",
                "load_schema_reference",
                "load_validation_run",
                "verify_execution_unchanged",
                "validation_input_files",
            },
        ),
    ):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        assert {
            node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        } == expected
    for name in ("DatasetValidationSpec", "ConversionChannelRule", "ConversionValidationSpec"):
        assert not hasattr(sys.modules[DatasetSchema.__module__], name)
    for name in ("load_validation_schema", "load_validation_run", "load_conversion_run"):
        assert not hasattr(schema_parser, name)


def test_public_dataset_entry_points_resolve_to_their_implementations():
    from pipelines import dataset
    from pipelines.dataset import append, compare, create, validate
    from pipelines.dataset.old import conversion_validation

    assert not hasattr(dataset, "update_zarr")
    assert not hasattr(dataset, "validate_standard_dataset")
    for module, name in (
        (create, "create_zarr"),
        (append, "append_zarr"),
        (validate, "validate_zarr"),
        (compare, "compare_zarr"),
        (conversion_validation, "validate_standard_conversion"),
        (conversion_validation, "run_conversion_validation"),
    ):
        assert getattr(dataset, name) is getattr(module, name)


def test_workflow_modules_do_not_import_model_dependencies():
    code = """
import sys
from pipelines.dataset import validate as dataset
from pipelines.dataset.old import conversion_validation as conversion
from pipelines.dataset import reports
from datakit.schema import DatasetSchema
from pipelines.dataset.configuration import DatasetValidationSpec
assert not hasattr(DatasetSchema, 'production_evidence')
assert not hasattr(DatasetSchema, 'consumer_evidence')
assert not hasattr(DatasetSchema, 'consumer_required')
assert not hasattr(DatasetSchema, 'conversion_evidence')
assert hasattr(DatasetValidationSpec, 'production_evidence')
assert not hasattr(DatasetValidationSpec, 'consumer_evidence')
assert 'pipelines.dataset.old.channel_metadata' not in sys.modules
assert 'pipelines.dataset.old.raw_truth' not in sys.modules
assert 'torch' not in sys.modules
assert 'xmetai' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
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
from pipelines import dataset
assert not hasattr(dataset, 'check_core_consumer')
assert len(dataset.__all__) == 13
for name in ('validate_standard_conversion',):
    assert callable(getattr(dataset, name))
    assert getattr(dataset, name).__module__ == 'pipelines.dataset.old.conversion_validation'
for name in ('era5_nc_to_zarr', 'oras5_nc_to_zarr_88'):
    importlib.import_module('examples.dataset_building.history.' + name)
truth = importlib.import_module('examples.dataset_processing.era5_extract_raw_truth_day')
assert len(truth.subset_channels('core-70')) == 70
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for path in (
        "pipelines/dataset/acceptance/consumer.py",
        "examples/dataset_building/check_core_consumer.py",
        "examples/dataset_building/configs/core_land.json",
    ):
        assert not (ROOT / path).exists()


def test_combination_workflow_is_not_exposed_or_imported():
    code = """
import importlib.util
import sys
from pipelines import dataset
from pipelines.dataset import validate as dataset_workflow
from pipelines.dataset.old import conversion_validation as conversion
from datakit.validation import structure as checks
for module in (dataset, dataset_workflow, conversion):
    assert not hasattr(module, 'validate_standard_combination')
    assert not hasattr(module, 'run_combination_validation')
assert not hasattr(checks, 'check_standard_time_alignment')
assert importlib.util.find_spec('pipelines.dataset.combination') is None
assert not any(name.endswith('.combination') for name in sys.modules)
assert set(dataset.__all__) == {
    'validate_standard_conversion', 'run_conversion_validation',
    'build_zarr', 'BuildReport',
    'create_zarr', 'append_zarr', 'validate_zarr', 'compare_zarr',
    'convert_to_zarr', 'extract_zarr_to_netcdf', 'extract_raw_truth_day',
    'split_netcdf_by_time_and_channel', 'validate_zarr_netcdf_consistency',
}
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    directory = ROOT / "examples" / "dataset_building"
    assert not (directory / "validate_standard_combination.py").exists()
    assert not (directory / "configs" / "s2s_land_combination.json").exists()


def test_production_definitions_do_not_load_acceptance():
    code = """
import sys
from pipelines.dataset.old import raw_truth as _raw_truth
from pipelines.dataset.old import channel_metadata
assert not any(
    name.startswith('pipelines.dataset.' + part)
    for part in ('checks', 'workflows', 'support') for name in sys.modules
)
assert 'torch' not in sys.modules
assert 'xmetai' not in sys.modules
assert not hasattr(_raw_truth, 'PRODUCT')
assert len(_raw_truth.CHANNELS) == 116
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_acceptance_implementation_has_one_workflow_local_home():
    assert not any((ROOT / "datakit").rglob("standard_dataset.py"))
    assert not (ROOT / "datakit" / "schema" / "loading.py").exists()
    for name in (
        "_standard_auxiliary.py",
        "_standard_quality.py",
        "standard_comparison.py",
        "evidence.py",
        "reports.py",
    ):
        assert not (ROOT / "datakit" / "validation" / name).exists()
    directory = ROOT / "pipelines" / "dataset"
    assert not list((directory / "definitions").glob("*.py"))
    assert not list((directory / "validation").glob("*.py"))
    assert {path.name for path in directory.glob("*.py")} == EXPECTED_PIPELINE_FILES

    legacy_sources = list((ROOT / "pipelines" / "dataset_building").rglob("*.py"))
    assert not legacy_sources, f"旧目录仍有源码未迁移：{legacy_sources}"
    assert not list((ROOT / "datakit" / "schema" / "datasets").glob("*.yaml"))
    assert len(list((ROOT / "config/dataset_specs").glob("*.yaml"))) == 10


def test_acceptance_subpackages_have_one_implementation_per_role():
    directory = ROOT / "pipelines" / "dataset"
    assert {path.name for path in directory.glob("*.py")} == EXPECTED_PIPELINE_FILES
    assert {p.name for p in (directory / "old").glob("*.py")} == {
        "__init__.py",
        "conversion.py",
        "channel_metadata.py",
        "netcdf_comparison.py",
        "raw_truth.py",
        "split.py",
        "conversion_validation.py",
    }
    for old in ("checks", "workflows", "support", "production"):
        assert not list((directory / old).rglob("*.py"))
    assert not (directory / "zarr_dataset.py").exists()
    assert not (directory / "specifications.py").exists()
    assert not (ROOT / "datakit/validation/schema.py").exists()
    code = """
import sys
from datakit.validation import structure, values, storage, comparison, check_result
assert check_result.CheckResult.__module__ == 'datakit.validation.check_result'
assert not any(name.startswith('pipelines.') for name in sys.modules)
from pipelines.dataset import reports, evidence, configuration, validate
for module, names in (
    (reports, ('ValidationReport', 'ensure_report_destination')),
    (configuration, ('DatasetValidationSpec', 'load_validation_schema', 'load_validation_run')),
    (structure, ('check_dataset_format', 'check_auxiliary_format', 'check_channel_semantics')),
    (values, ('StandardQualityAccumulator', 'check_auxiliary_values', 'check_statistics')),
    (storage, ('check_root_metadata', 'check_array_encoding')),
    (evidence, ('check_production_receipt', 'attach_conversion_evidence')),
    (comparison, ('compare_datasets', 'compare_channel_values')),
    (validate, ('scan_standard_blocks',)),
):
    assert all(callable(getattr(module, name)) for name in names)
assert not hasattr(reports, 'attach_conversion_evidence')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_validation_examples_only_adapt_cli_arguments():
    directory = ROOT / "examples" / "dataset_building"
    for name in ("history/validate_standard_conversion.py",):
        tree = ast.parse((directory / name).read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(
            module.split(".")[0] in {"numpy", "xarray", "zarr", "torch", "xmetai"}
            for module in imports
        ), name


def test_base_validation_does_not_depend_on_pipeline_or_examples():
    for path in (ROOT / "datakit/validation").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                name.split(".")[0] in {"pipelines", "examples", "torch", "xmetai"} for name in names
            ), (path, node.lineno)


def test_configuration_and_reports_do_not_import_operations():
    from importlib.util import resolve_name

    forbidden = {
        "create",
        "append",
        "validate",
        "compare",
        "conversion_validation",
        "_channel_metadata",
        "_raw_truth",
    }
    for name in ("configuration.py", "reports.py"):
        path = ROOT / "pipelines/dataset" / name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = (
                resolve_name("." * node.level + (node.module or ""), "pipelines.dataset")
                if node.level
                else node.module or ""
            )
            assert not any(module == "pipelines.dataset." + part for part in forbidden), (
                path,
                node.lineno,
            )


def test_check_rules_do_not_open_files_or_walk_directories():
    forbidden = {
        "open",
        "open_zarr",
        "open_group",
        "open_array",
        "open_dataset",
        "read_bytes",
        "read_text",
        "iterdir",
        "rglob",
        "glob",
        "stat",
        "is_file",
        "write_text",
        "write_bytes",
        "save",
        "load",
    }
    directory = ROOT / "datakit" / "validation"
    for path in (directory / "structure.py", directory / "values.py", directory / "storage.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            assert name not in forbidden, (path, node.lineno, name)


def test_core_operations_do_not_import_legacy_workflows():
    code = """
import sys
from pipelines.dataset import create, append, compare, validate
assert not any(n == 'pipelines.dataset.old' or n.startswith('pipelines.dataset.old.') for n in sys.modules)
from pipelines.dataset import configuration
assert not hasattr(configuration, 'ConversionChannelRule')
assert not hasattr(configuration, 'ConversionValidationSpec')
assert not hasattr(configuration, 'load_conversion_run')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
