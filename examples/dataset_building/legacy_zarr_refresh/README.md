# Legacy Zarr refresh example

This directory is a self-contained dataset-building example. It is intentionally
separate from both `era5_nc_to_zarr.py` and the supported `datakit` package API.
It refreshes an existing normalized Zarr store; it does not convert raw ERA5
NetCDF files and it does not infer new physical unit conversions.

The modules cover source inspection, channel/coordinate alignment, Zarr v2
writing, value validation and optional `const`/`weight`/`mask` migration. The
JSON file keeps server paths and dataset-specific rules outside Python code.

## Commands

Copy `config.example.json`, replace the placeholder paths, then inspect and
dry-run before writing:

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json inspect --dataset all

python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json pilot --dataset oras5 \
  --output-root /path/to/output --count 4 --dry-run
```

Run a pilot and then the complete conversion:

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json pilot --dataset oras5 \
  --output-root /path/to/output --count 4 --full-data-check

python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json convert --dataset oras5 \
  --output-root /path/to/output --workers 4
```

Validate an existing output independently:

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json validate --dataset oras5 \
  --zarr /path/to/output/oras5.refreshed.zarr --full-data-check
```

Migrate auxiliary files only after the target Zarr is complete:

```bash
python examples/dataset_building/legacy_zarr_refresh/cli.py \
  --config /path/to/datasets.json auxiliary --dataset oras5 \
  --output-root /path/to/output --target-name oras5.refreshed.zarr --dry-run
```

The embedded Zarr mask is preserved by default. Replacing it from
`const.nc/lsm` requires `--refresh-embedded-mask` after the two mask semantics
have been verified. `const.nc` preserves its legacy channels and order; the
migration does not invent additional channels.

## Test data

Default tests generate deterministic, very small Zarr and NetCDF inputs under
pytest's temporary directory. They execute the complete write/read workflow and
cover channel reordering, latitude reversal, statistics correction, masks,
Zarr v2 dot-separated chunk keys, validation manifests and auxiliary-file
alignment. The generated values are deliberately simple so expected results can
be asserted exactly.

No server data is required by the default test suite and no dataset fixture is
committed to Git. The only skipped-by-default test is the external-store
contract check. Set `XMETAI_REFRESH_TEST_STORE` to a completed server output and
run the integration-marked test explicitly when server access is available.

A missing server path never skips the generated end-to-end or auxiliary tests.
Those tests are the required CI gate and must pass locally and in GitHub Actions.

Zarr 3.1.6 or newer is required to read the server's fixed-length string arrays.
