# Dataset-building examples

This directory contains separate, runnable dataset-building workflows:

- `era5_nc_to_zarr.py` builds a normalized Zarr dataset from ERA5 NetCDF files.
- `oras5_nc_to_zarr_88.py` builds the 88-channel ORAS5 dataset.
- [`legacy_zarr_refresh.py`](legacy_zarr_refresh.py) contains the single-file legacy Zarr refresh,
  validation and auxiliary-file migration example.

## Legacy Zarr refresh

Like the NetCDF conversion examples, the refresh example uses argument parsing,
helper functions and a `main()` entry point in one Python file. It refreshes
existing normalized Zarr stores, not raw NetCDF inputs, and does not apply
physical unit conversions again. Its workflow deliberately remains outside
the supported `datakit` API while the legacy dataset contract is under review.

All five operations are preserved: source inspection, pilot refresh, complete
refresh, output validation and optional auxiliary-file migration. Channel
metadata, source adapters, Zarr v2 writing and validation now live in the same
script rather than a separate package. Server paths and per-dataset correction
rules remain in external JSON, not hardcoded Python defaults.

Copy [the configuration template](configs/legacy_zarr_refresh.example.json),
replace its placeholder paths, then inspect and dry-run before writing:

```bash
python examples/dataset_building/legacy_zarr_refresh.py \
  --config /path/to/datasets.json inspect --dataset all

python examples/dataset_building/legacy_zarr_refresh.py \
  --config /path/to/datasets.json pilot --dataset oras5 \
  --output-root /path/to/output --count 4 --dry-run
```

Write a pilot, then refresh the complete dataset:

```bash
python examples/dataset_building/legacy_zarr_refresh.py \
  --config /path/to/datasets.json pilot --dataset oras5 \
  --output-root /path/to/output --count 4 --full-data-check

python examples/dataset_building/legacy_zarr_refresh.py \
  --config /path/to/datasets.json convert --dataset oras5 \
  --output-root /path/to/output --workers 4
```

Writes use staging and a resumable progress record. Existing completed outputs
are not deleted automatically; explicit `--overwrite` backs them up before
replacement. The input remains read-only.

Validate an output or migrate its auxiliary files:

```bash
python examples/dataset_building/legacy_zarr_refresh.py \
  --config /path/to/datasets.json validate --dataset oras5 \
  --zarr /path/to/output/oras5.refreshed.zarr --full-data-check

python examples/dataset_building/legacy_zarr_refresh.py \
  --config /path/to/datasets.json auxiliary --dataset oras5 \
  --output-root /path/to/output --target-name oras5.refreshed.zarr --dry-run
```

Remove `--dry-run` to write auxiliary files. The embedded Zarr mask is preserved
unless `--refresh-embedded-mask` is explicitly requested after verifying its
semantics against `const.nc/lsm`. Constant channels retain their original names
and order; migration does not invent additional channels.

The old `legacy_zarr_refresh/cli.py` entry point and package submodules have
been removed. Use the single script above; from the repository root,
`python -m examples.dataset_building.legacy_zarr_refresh` also works.
Tests import helpers directly from the single module.

Regression tests in `tests/dataset_building/legacy_zarr_refresh/` generate small,
deterministic Zarr and NetCDF inputs and cover value alignment, statistics,
masks, auxiliary migration, CLI operations and interrupted-write resumption.
Only the external-store contract check requires `XMETAI_REFRESH_TEST_STORE`;
missing server data does not skip synthetic regression tests. Zarr 3.1.6 or
newer is required to read the server's fixed-length string arrays.

## Standard dataset validation

Fixed standard models and strict YAML parsing live in
[`datakit/schema/datasets`](../../datakit/schema/datasets), alongside the ten YAML standards.
Checks, scanning, runtime binding, evidence IO, batching and reporting live in
[`pipelines/dataset_building`](../../pipelines/dataset_building).
The former acceptance package is removed. Four sibling packages separate
production definitions, checks, workflows and support.
`datasets/specifications.py` defines fixed models, constraints and derived expectations.
`datasets/inputs.py` loads strict YAML; `support/specifications.py` defines runtime
and conversion specifications, while `support/inputs.py` binds execution JSON and receipts.
`datakit.schema.datasets.load_schema` returns a target schema, `load_validation_schema` creates its
unbound execution specification, and `load_validation_run` binds execution JSON.
The `checks` package groups format, values, storage and conversion rules;
auxiliary format and values follow the same separation as main data.
`workflows/dataset.py` owns quick/full scanning, resource control and batching.
`workflows/conversion.py` runs source-to-target fidelity comparisons, not conversion.
`support/reports.py` owns results, completion summaries and persistence.
Conversion receipt validation belongs to `checks/conversion.py`, not report output.
Existing source mappings and `channel_metadata.py` remain in `production`;
validation never uses their generated values as its truth.
This directory only adapts CLI arguments, signals and exit codes:

- `validate_standard_dataset.py`: check one or more independent YAML standards.
- `validate_standard_conversion.py`: compare a frozen source and target using explicit rules.
- [`datakit/schema/datasets/`](../../datakit/schema/datasets/): ten packaged YAML standards.
- `configs/`: conversion and legacy-refresh templates.

Run from the repository root:

```bash
python -m examples.dataset_building.validate_standard_dataset \
  --schema-directory datakit/schema/datasets \
  --mode quick --sample-blocks 3 --output /path/outside/data/reports
```

For one dataset, use `--schema datakit/schema/datasets/era5_land_c8_1p5_h24.yaml`.
Multiple paths may follow `--schema`; do not combine multi-dataset selection with
a single `--dataset-path` override. Old profile/inventory options are removed.
Conversion JSON configurations reference YAML through `target_schema`. Relative paths
are resolved against the containing configuration, including evidence paths.
Reports and independent evidence remain JSON. A quick pass verifies the
selected main-data blocks and complete metadata/coordinate checks. A full scan
requires `--mode full` without `--max-blocks`, plus the applicable physical
coverage and production completion evidence. See the
[validation guide](../../docs/DATASET_VALIDATION.md) for report fields and tests.

For a frozen run, use `--run-config path/to/run.json` instead of `--schema`.
This JSON references the unchanged YAML and supplies an `execution` object with
`snapshot_id`, receipt `{path, sha256}` references and applicable requirements.
Runtime production/conversion receipts are not accepted inside YAML.
Adding a receipt to execution JSON does not change the YAML hash bound by it.
Conversion configurations also accept this
nested `execution` object. A run cannot disable a schema-required evidence gate.

Core reader probes and consumer/model receipt checks have been removed. The
two validation CLIs cover data format, integrity, quality and conversion fidelity,
not model compatibility. Retired
`consumer_required`, `consumer_evidence` and `model_evidence` settings are rejected.
JSON/CSV/Markdown reports no longer expose a consumer-compatibility conclusion.

Combination validation, its CLI and configuration template have also been removed.
Selecting multiple YAMLs or run configurations still validates each dataset
independently; it does not impose shared time, grid or mask requirements or
produce a joint-use conclusion. Tool version `1.8` removes the
`cross_dataset_compatibility` report category and `combination.scope` check.
Historical reports are not rewritten.

Program regression tests live in `tests/dataset_building/acceptance/`; they use small fixtures
and injected faults. Passing these tests does not establish that server data
has passed validation.
