"""Diamond 站点文本与站点气候概率表 ↔ Xarray Dataset。

Diamond 输出：(time, station)，包含经纬度、海拔、降水和额外数值列。
概率表输出：(cycle_time, station, threshold)，cycle_time 保留 MMDDHH 标签。
文本立即加载；保留源数值和缺测哨兵，不做质量控制或时区转换。
写回需要读取时保留的文件头、站点顺序等元数据。
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .._layout import templates


def read_records(records, description, **options):
    """读取已发现的文件记录，供统一入口和逐文件处理复用。"""
    stations = options.pop("stations", None)
    fixed_metadata = options.pop("fixed_station_metadata", False)
    tables = []
    if options:
        raise TypeError(f"Text table options are not supported: {sorted(options)}")
    probability = "thresholds" in description
    file_format = description.get("format", {})
    columns = file_format.get(
        "columns", ["station", "lon", "lat", "altitude", "precipitation"]
    )
    frames, manifest = [], []
    for record in records:
        skip = file_format.get("header_lines", 0 if probability else 2)
        with record.path.open(encoding=file_format.get("encoding", "gbk")) as stream:
            headers = [stream.readline() for _ in range(skip)]
            rows = np.loadtxt(stream, ndmin=2)
        minimum = len(description["thresholds"]) + 1 if probability else len(columns)
        if (
            rows.size == 0
            or rows.shape[1] < minimum
            or (probability and rows.shape[1] != minimum)
        ):
            raise ValueError(f"Invalid station table shape {rows.shape}: {record.path}")
        if stations is not None:
            rows = rows[np.isin(rows[:, 0], stations)]
            if not len(rows):
                continue
        station = rows[:, 0]
        if (
            not np.all(np.isfinite(station))
            or not np.all(station == station.astype("i8"))
            or len(set(station)) != len(station)
        ):
            raise ValueError(f"Station IDs must be unique integers: {record.path}")
        if fixed_metadata and not probability:
            canonical = [
                columns.index(name)
                for name in ("station", "lon", "lat", "altitude", "precipitation")
            ]
            tables.append((record, headers, rows[:, canonical]))
            continue
        if probability:
            dataset = xr.Dataset(
                {"probability": (("station", "threshold"), rows[:, 1:])},
                coords={
                    "station": station.astype("i8"),
                    "threshold": description["thresholds"],
                },
            )
            axis = "cycle_time"
        else:
            names = columns[1:] + [
                f"column_{i}" for i in range(len(columns), rows.shape[1])
            ]
            dataset = xr.Dataset(
                {name: ("station", rows[:, i + 1]) for i, name in enumerate(names)},
                coords={"station": station.astype("i8")},
            )
            axis = "time"
        dataset = dataset.expand_dims({axis: [record.fields[axis]]})
        frames.append(dataset)
        manifest.append(
            {
                "fields": record.fields,
                "headers": headers,
                "stations": station.astype("i8"),
                "columns": list(dataset.data_vars),
            }
        )
    if fixed_metadata and not probability:
        return _station_collection(tables)
    if not frames:
        raise ValueError("No valid station data parsed")
    labels = [frame[axis].values[0] for frame in frames]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate {axis} records")
    result = xr.concat(
        frames, dim=axis, join="outer", coords="minimal", compat="equals"
    ).sortby(axis)
    result.encoding["dataset_io_manifest"] = manifest
    return result


def write(dataset, root, description, **options):
    """按当前布局写出 Dataset，返回输出文件路径列表。"""
    if options:
        raise TypeError(
            f"Text table write options are not supported: {sorted(options)}"
        )
    patterns = templates(description)
    if len(patterns) != 1:
        raise ValueError("Text writing requires one reversible filename template")
    probability = "thresholds" in description
    manifest = dataset.encoding.get("dataset_io_manifest")
    if not manifest:
        raise NotImplementedError(
            "Text writing requires headers and record membership retained by read()"
        )
    plans = []
    planned_paths = set()
    axis = "cycle_time" if probability else "time"
    for record in manifest:
        path = root / patterns[0][1].render(record["fields"])
        if path.exists() or path in planned_paths:
            raise FileExistsError(path)
        part = dataset.sel(
            {axis: record["fields"][axis], "station": record["stations"]}
        )
        values = (
            part.probability.transpose("station", "threshold").values
            if probability
            else np.column_stack([part[name].values for name in record["columns"]])
        )
        rows = np.column_stack([record["stations"], values])
        planned_paths.add(path)
        plans.append((path, record["headers"], rows))
    for path, headers, rows in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(
            "x", encoding=description.get("format", {}).get("encoding", "gbk")
        ) as stream:
            stream.writelines(headers)
            np.savetxt(stream, rows, fmt="%.17g")
    return [str(p) for p, *_ in plans]


def parse_station_file_time(path: Path) -> datetime:
    """解析 Diamond 站点文件的观测时刻。

    支持两种命名：``station_dir/YYYYMMDDHH.000`` 与 ``station_dir/YYYYMMDD/HH.000``。
    """
    path = Path(path)
    name = path.stem
    if len(name) == 10 and name.isdigit():
        return datetime.strptime(name, "%Y%m%d%H")  # noqa: DTZ007 -- preserve source wall time
    if len(name) == 2 and name.isdigit() and path.parent.name.isdigit():
        return datetime.strptime(path.parent.name + name, "%Y%m%d%H")  # noqa: DTZ007 -- preserve source wall time
    raise ValueError(f"无法解析观测文件时间: {path}")


def _station_collection(tables):
    """Assemble bounded station columns; retain metadata at first appearance."""
    if not tables:
        raise ValueError("No valid station data parsed")
    tables.sort(key=lambda item: item[0].fields["time"])
    stations = np.unique(
        np.concatenate([rows[:, 0].astype("i8") for _, _, rows in tables])
    )
    times = [record.fields["time"] for record, _, _ in tables]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate time records")
    precipitation = np.full((len(tables), len(stations)), np.nan, dtype="f8")
    metadata = np.full((3, len(stations)), np.nan, dtype="f8")
    filled = np.zeros(len(stations), dtype=bool)
    for position, (_, _, rows) in enumerate(tables):
        indexes = np.searchsorted(stations, rows[:, 0].astype("i8"))
        precipitation[position, indexes] = rows[:, 4]
        first = ~filled[indexes]
        metadata[:, indexes[first]] = rows[first, 1:4].T
        filled[indexes] = True
    return xr.Dataset(
        {"precipitation": (("time", "station"), precipitation)},
        coords={
            "time": times,
            "station": stations,
            "lat": ("station", metadata[1]),
            "lon": ("station", metadata[0]),
            "altitude": ("station", metadata[2]),
        },
    )
