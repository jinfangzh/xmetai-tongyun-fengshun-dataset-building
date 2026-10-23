"""BABJ 台风报文：一个 GBK 文本文件保存一条台风的多个时次和时效。

    root/
      babj2501.dat
      babj2502.dat

每个文件前三行为报文头；第二行提供台风名和编号。
数值列依次为：年、月、日、时、预报时效、经度、纬度、最低气压、最大风速。
多余数值列原样保留，时间保持源数据的北京时间。
read() 返回 record 维的原始记录，保留全部时效；是否只取 000 时效由调用方决定。
写回需要读取时保留的表头、文件名和记录信息。不声明额外变量映射。
"""

import sys

import numpy as np
import xarray as xr

from datakit.io import open_dataset

DESCRIPTION = {
    "name": "babj",
    "format": {
        "encoding": "gbk",
        "header_lines": 3,
        "columns": [
            "year",
            "month",
            "day",
            "hour",
            "lead_time",
            "lon",
            "lat",
            "pmin",
            "vmax",
        ],
    },
    "layout": {"pattern": "babj*.dat"},
    "time_zone": "Asia/Shanghai",
    "example": "babj2501.dat",
}


def open(path, *, selection=None):
    """配置原始 BABJ 报文访问，保留全部时效。"""
    return open_dataset(
        path, DESCRIPTION, selection=selection, backend=sys.modules[__name__]
    )


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
    if options:
        raise TypeError(f"BABJ write options are not supported: {sorted(options)}")
    file_format = description["format"]
    manifest = dataset.encoding.get("dataset_io_manifest")
    if not manifest:
        raise NotImplementedError(
            "BABJ output requires source headers retained by read()"
        )
    paths = [root / record["filename"] for record in manifest]
    if len(set(paths)) != len(paths) or any(p.exists() for p in paths):
        raise FileExistsError("BABJ output filenames collide or already exist")
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding=file_format["encoding"]) as stream:
            stream.writelines(item["headers"])
            np.savetxt(stream, values, fmt="%.17g")
    return list(map(str, paths))
