# Dataset-building examples

This directory contains separate, runnable dataset-building workflows:

- `era5_nc_to_zarr.py` builds a normalized Zarr dataset from ERA5 NetCDF files.
- `oras5_nc_to_zarr_88.py` builds the 88-channel ORAS5 dataset.
- `legacy_zarr_refresh/` contains the self-contained legacy Zarr refresh,
  validation and auxiliary-file migration example.

The refresh workflow deliberately lives outside the `datakit` package while
its dataset contract is under review. See
[`legacy_zarr_refresh/README.md`](legacy_zarr_refresh/README.md) for commands,
scope and test-data policy.

## Standard dataset validation

Product definitions, checks, scanning, configuration loading and report writing
live in [`pipelines/dataset_building`](../../pipelines/dataset_building/README.md).
This directory provides executable entry points and their configurations:

- `validate_standard_dataset.py`: check one profile or the ten-profile inventory.
- `validate_standard_conversion.py`: compare a frozen source and target using explicit rules.
- `validate_standard_combination.py`: check compatibility between dataset members.
- `check_core_consumer.py`: run the separate core consumer probe.
- `configs/profiles/`: ten batch profiles and `inventory.json`.
- `configs/`: conversion, combination and consumer probe templates.

Run from the repository root:

```bash
python -m examples.dataset_building.validate_standard_dataset \
  --inventory examples/dataset_building/configs/profiles/inventory.json \
  --mode quick --sample-blocks 3 --output /path/outside/data/reports
```

The configurations retain their existing JSON format. A quick pass verifies the
selected main-data blocks and complete metadata/coordinate checks. A full scan
requires `--mode full` without `--max-blocks`, plus the applicable physical
coverage and production completion evidence. See the
[validation guide](../../docs/DATASET_VALIDATION.md) for report fields and tests.

Program regression tests remain in `tests/validation/`; they use small fixtures
and injected faults. Passing these tests does not establish that server data
has passed validation.
