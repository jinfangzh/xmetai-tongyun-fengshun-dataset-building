# Dataset-building examples

This directory contains separate, runnable dataset-building workflows:

- `era5_nc_to_zarr.py` builds a normalized Zarr dataset from ERA5 NetCDF files.
- `oras5_nc_to_zarr_88.py` builds the 88-channel ORAS5 dataset.
- [`legacy_zarr_refresh.py`](legacy_zarr_refresh.py) contains the single-file legacy Zarr refresh,
  validation and auxiliary-file migration example.

## Legacy Zarr refresh

The single-file example reads existing normalized v2 or v3 sources and writes
Zarr v3. It does not reapply unit conversions or normalize the main values again.
Source paths, explicit aliases, statistics corrections and mask adapters live in
JSON; `target_schema` selects an independent YAML standard in
[`datakit/schema/datasets`](../../datakit/schema/datasets).

The [JSON example](configs/legacy_zarr_refresh.example.json) contains all ten
product/variant selections. Replace the placeholder input paths and mask source
for the selected dataset; verify statistics corrections independently before
production. Empty correction dictionaries do not certify the source statistics.
Relative input and YAML paths resolve against the JSON file's directory. If
copying the JSON elsewhere, update `target_schema`, preferably to an absolute path.
Install the repository and its dependencies before running the example from
another directory.

The target YAML supplies the exact channel order, grid and latitude direction,
root identity, schema/content versions, data revision and channel semantics.
CRA c71 is south-to-north; the other nine are north-to-south. Missing or extra
channels fail, even if the total count matches. Data, statistics and embedded
masks are reordered together. Source time coverage, cadence, radiation window
and full grid must agree with the selected standard before output is created.
For appended data, prepare a reviewed YAML with the actual coverage and revision;
the refresh never changes a standard to fit its source.

Inspect and dry-run before writing:

```bash
python -m examples.dataset_building.legacy_zarr_refresh \
  --config /path/to/datasets.json inspect --dataset oras5_c88_1p5_h24

python -m examples.dataset_building.legacy_zarr_refresh \
  --config /path/to/datasets.json pilot --dataset oras5_c88_1p5_h24 \
  --output-root /workspace/xujn/refresh-pilot --count 4 --dry-run
```

Write a pilot, then refresh to a separate output directory:

```bash
python -m examples.dataset_building.legacy_zarr_refresh \
  --config /path/to/datasets.json pilot --dataset oras5_c88_1p5_h24 \
  --output-root /workspace/xujn/refresh-pilot --count 4 --full-data-check

python -m examples.dataset_building.legacy_zarr_refresh \
  --config /path/to/datasets.json convert --dataset oras5_c88_1p5_h24 \
  --output-root /workspace/xujn/refresh-output --workers 1 --full-data-check
```

Each output main chunk contains one full frame in float16. Time is stored as
`datetime64[ns]`; strings use fixed-length UTF-32, and v3 default chunk keys use
`/` separators. Arrays carry native `dimension_names`; consolidated metadata is
written inline. All-fill frames are physically written rather than silently
omitted. At 0.25 degrees, a c116 frame alone is about 230 MiB; increase workers
only after budgeting for source reads, decoding, reordering and compression.

Writes use staging and a resumable progress record bound to the YAML SHA256.
Changing the template invalidates old staging. Existing completed outputs require
explicit `--overwrite`, which keeps a backup. The input remains read-only.
A pilot retains the selected target identity but covers only its chosen time
slice; it must not be presented as satisfying the full production coverage.

Validate a complete refreshed output against its source, then migrate auxiliaries:

```bash
python -m examples.dataset_building.legacy_zarr_refresh \
  --config /path/to/datasets.json validate --dataset oras5_c88_1p5_h24 \
  --zarr /workspace/xujn/refresh-output/<output_name> --full-data-check

python -m examples.dataset_building.legacy_zarr_refresh \
  --config /path/to/datasets.json auxiliary --dataset oras5_c88_1p5_h24 \
  --output-root /workspace/xujn/refresh-output --strict --dry-run
```

Replace `<output_name>` with the selected JSON entry's directory name. Remove
`--dry-run` to write auxiliaries. Their axes follow the output grid. With a target
schema, const channel sets must match and are reordered to the declared order;
NetCDF data-variable names are preserved without inventing a naming standard.
Weights are written as `weight.nc` for ORAS5 and `weight.npy` for the other
products. The declared nonnegative tolerance is checked without clipping values.
The embedded mask is preserved unless `--refresh-embedded-mask` is explicitly
requested after verifying its semantics. Optional external masks do not acquire
an approved role simply by being copied.

Execution manifests are siblings of the Zarr directory:

```text
refresh-output/
├── <output_name>/
├── <output_name>.refresh_manifest.json
└── <output_name>.auxiliary_refresh_manifest.json
```

This keeps unregistered JSON reports out of the standard store. `read_manifest`
reads the new location and can still read historical in-store manifests. A
refresh manifest is a conversion diagnostic, not an independent production
completion receipt. Run the standard validation CLI separately after auxiliary
migration; `--full-data-check` does not replace its quality or publication gates.

Regression tests cover known-value samples, both source formats, exact product
orders, native v3 encoding, auxiliary formats, CLI operations, missing chunks and
interrupted-write resumption. Ten explicitly miniature product samples retain
all 786 ordered channel entries and product semantics while reducing only the
test grid and coverage. They are not production datasets. Only the external-store
check needs `XMETAI_REFRESH_TEST_STORE`; synthetic tests do not require a server.

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
