"""BABJ diamond7 报文 ↔ 原始记录形式的 Xarray Dataset。

输入：GBK 报文，文件头提供台风名与编号，数值行含年月日时、时效及气象字段。
输出：record 维，保留全部时效和额外数值列；不筛选分析记录、不转换时间。
写回需要 read 保留的报文头与文件名。轨迹筛选与评估属于上层流程。
"""

import numpy as np
import xarray as xr


def read_records(records, description, **options):
    """读取已发现的文件记录，供统一入口和逐文件处理复用。"""
    if options:
        raise TypeError(f"BABJ options are not supported: {sorted(options)}")
    parts, manifest = [], []
    offset = 0
    for record in records:
        with record.path.open(encoding="gbk") as stream:
            headers = [stream.readline() for _ in range(3)]
            tokens = headers[1].split()
            if len(tokens) < 2:
                raise ValueError(f"Invalid BABJ header: {record.path}")
            rows = np.loadtxt(stream, ndmin=2)
        if rows.size == 0 or rows.shape[1] < 9:
            raise ValueError(f"Invalid BABJ numeric table: {record.path}")
        names = [
            "year",
            "month",
            "day",
            "hour",
            "lead_time",
            "lon",
            "lat",
            "pmin",
            "vmax",
        ]
        names += [f"column_{i}" for i in range(9, rows.shape[1])]
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
        with path.open("x", encoding="gbk") as stream:
            stream.writelines(item["headers"])
            np.savetxt(stream, values, fmt="%.17g")
    return list(map(str, paths))


def assemble_storm_tracks(storms):
    """Preserve source wall times and missing positions across storm tracks."""
    if not storms:
        raise ValueError("No decoded storm analyses")
    times = sorted({t for storm in storms for t in storm["analyses"]})
    time_index = np.array(times, dtype="datetime64[ns]")
    tcid_index = np.array([storm["tcid"] for storm in storms], dtype=object)
    tcname_index = np.array([storm["tcname"] for storm in storms], dtype=object)

    shape = (len(times), len(storms))
    lat = np.full(shape, np.nan, dtype="f8")
    lon = np.full(shape, np.nan, dtype="f8")
    pmin = np.full(shape, np.nan, dtype="f8")
    vmax = np.full(shape, np.nan, dtype="f8")
    slot = {t: i for i, t in enumerate(times)}
    for j, storm in enumerate(storms):
        for moment, record in storm["analyses"].items():
            i = slot[moment]  # 该时刻必有槽位（times 是全并集）
            lat[i, j] = record["lat"]
            lon[i, j] = record["lon"]
            pmin[i, j] = record["pmin_hpa"]
            vmax[i, j] = record["vmax_ms"]

    ds = xr.Dataset(
        {
            "lat": (("time", "storm"), lat),
            "lon": (("time", "storm"), lon),
            "pmin": (("time", "storm"), pmin),
            "vmax": (("time", "storm"), vmax),
        },
        coords={
            "time": time_index,
            "storm": tcid_index,
            "tcname": ("storm", tcname_index),
        },
    )
    ds["lat"].attrs.update(units="degrees_north", long_name="Storm centre latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="Storm centre longitude")
    ds["pmin"].attrs.update(units="hPa", long_name="Minimum sea level pressure")
    ds["vmax"].attrs.update(units="m/s", long_name="Maximum sustained wind")

    return ds
