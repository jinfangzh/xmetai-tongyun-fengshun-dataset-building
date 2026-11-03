"""Read and write BABJ track tables using caller-supplied format declarations."""

import numpy as np
import xarray as xr

from ._writing import check_outputs, staged_output, write_mode


def read_records(records, description, **options):
    """读取已发现的文件记录，供统一入口和逐文件处理复用。"""
    if options:
        raise TypeError(f"BABJ options are not supported: {sorted(options)}")
    file_format = description["format"]
    columns = file_format["columns"]
    parts, manifest = [], []
    offset = 0
    for record in records:
        with record.path.open(encoding=file_format["encoding"]) as stream:
            headers = [stream.readline() for _ in range(file_format["header_lines"])]
            tokens = headers[1].split()
            if len(tokens) < 2:
                raise ValueError(f"Invalid BABJ header: {record.path}")
            rows = np.loadtxt(stream, ndmin=2)
        if rows.size == 0 or rows.shape[1] < len(columns):
            raise ValueError(f"Invalid BABJ numeric table: {record.path}")
        names = list(columns)
        names += [f"column_{i}" for i in range(len(columns), rows.shape[1])]
        part = xr.Dataset(
            {name: ("record", rows[:, i]) for i, name in enumerate(names)},
            coords={
                "record": np.arange(offset, offset + len(rows)),
                "storm": ("record", [tokens[1]] * len(rows)),
                "tcname": ("record", [tokens[0]] * len(rows)),
            },
        )
        offset += len(rows)
        parts.append(part)
        manifest.append(
            {
                "filename": record.path.name,
                "headers": headers,
                "columns": names,
                "records": part.record.values,
            }
        )
    result = xr.concat(
        parts, dim="record", join="outer", coords="minimal", compat="equals"
    )
    result.encoding["dataset_io_manifest"] = manifest
    return result


def write(dataset, root, description, **options):
    """按当前布局写出 Dataset，返回输出文件路径列表。"""
    mode = write_mode(options)
    if options:
        raise TypeError(f"BABJ write options are not supported: {sorted(options)}")
    file_format = description["format"]
    manifest = dataset.encoding.get("dataset_io_manifest")
    if not manifest:
        raise NotImplementedError(
            "BABJ output requires source headers retained by read()"
        )
    paths = [root / record["filename"] for record in manifest]
    check_outputs(paths, mode)
    rows = [
        np.column_stack(
            [
                dataset[name].sel(record=item["records"]).values
                for name in item["columns"]
            ]
        )
        for item in manifest
    ]
    for path, item, values in zip(paths, manifest, rows):
        with staged_output(path, mode) as staged:
            with staged.open("x", encoding=file_format["encoding"]) as stream:
                stream.writelines(item["headers"])
                np.savetxt(stream, values, fmt="%.17g")
    return list(map(str, paths))
