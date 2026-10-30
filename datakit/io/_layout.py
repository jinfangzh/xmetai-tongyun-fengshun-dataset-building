"""Validate explicit file layouts and parse reversible relative path templates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Any


def source_config(name: str, *, directory: str | Path | None = None) -> Path:
    """Locate a named YAML declaration in an explicit, checkout or installed directory.

    Arbitrary external YAML paths can be passed directly to get_description() and
    open_dataset(). This helper only locates the shipped dataset declarations.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Source configuration name must be a simple filename stem")
    if directory is None:
        import sysconfig

        roots = [
            Path(__file__).resolve().parents[2] / "config" / "data_sources",
            Path(sysconfig.get_path("data")) / "config" / "data_sources",
        ]
    else:
        roots = [Path(directory).expanduser()]
    for root in roots:
        path = root / f"{name}.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Source configuration {name!r} not found in {roots}")


def get_description(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load an explicit declaration or return an independent copy."""
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
    else:
        import yaml

        candidate = Path(value).expanduser()
        if not candidate.is_file():
            raise ValueError(
                "IO requires an explicit layout mapping or YAML path; "
                "dataset declarations are YAML files in config/data_sources"
            )
        result = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("A dataset layout must be a mapping")
    return result


def get_layout(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate raw file rules without applying variable mappings."""
    result = get_description(value)
    result.pop("mapping", None)
    result.pop("contract", None)
    return load_layout(result)


def load_layout(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    result = get_description(value)
    required = {"name", "layout", "example"}
    if missing := required - result.keys():
        raise ValueError(f"Missing layout fields: {sorted(missing)}")
    allowed = required | {
        "format",
        "notes",
        "lead_time",
        "member",
        "structure",
        "cycle",
        "thresholds",
        "time_zone",
    }
    if unknown := result.keys() - allowed:
        raise ValueError(f"Unknown layout fields: {sorted(unknown)}")
    if isinstance(result.get("format"), str):
        result["format"] = {"name": result["format"]}
    if "format" in result and not isinstance(result["format"], dict):
        raise TypeError("format must be a format name or mapping")
    format_name = result.get("format", {}).get("name")
    if format_name is not None and (
        not isinstance(format_name, str)
        or format_name
        not in {
            "netcdf",
            "grib",
            "zarr",
            "numpy",
            "diamond_station",
            "station_probability",
            "babj",
        }
    ):
        raise ValueError(f"Unsupported dataset format: {format_name!r}")
    if not isinstance(result["layout"], dict):
        raise TypeError("layout must be a mapping")
    allowed_layout = {"directory", "filename", "files", "groups", "pattern"}
    if unknown := result["layout"].keys() - allowed_layout:
        raise ValueError(f"Unknown layout fields: {sorted(unknown)}")
    layout = result["layout"]
    if sum(key in layout for key in ("filename", "files", "pattern")) > 1:
        raise ValueError("Use one of layout.filename, files or pattern")
    if layout.get("directory") and not any(
        key in layout for key in ("filename", "files", "pattern")
    ):
        raise ValueError("A directory layout requires a filename, files or pattern")
    patterns = templates(result)
    if patterns and not any(
        template.parse(result["example"]) is not None for _, template in patterns
    ):
        raise ValueError("The layout example does not match its layout")
    return result


_DATE_PARTS = {
    "Y": (r"\d{4}", "year"),
    "m": (r"\d{2}", "month"),
    "d": (r"\d{2}", "day"),
    "H": (r"\d{2}", "hour"),
    "M": (r"\d{2}", "minute"),
    "S": (r"\d{2}", "second"),
}


class PathTemplate:
    """A reversible template; repeated date fields must agree across the path."""

    def __init__(self, template: str):
        if Path(template).is_absolute() or ".." in Path(template).parts:
            raise ValueError("Layout templates must stay relative to the dataset root")
        self.template = template
        self.fields: list[tuple[str, str, str]] = []
        self.parts = list(Formatter().parse(template))
        expression = ""
        for literal, field, spec, conversion in self.parts:
            expression += re.escape(literal)
            if field is None:
                continue
            if not field.isidentifier() or conversion:
                raise ValueError(f"Invalid template field: {field!r}")
            group = f"f{len(self.fields)}"
            if "%" in spec:
                pattern = ""
                cursor = 0
                for match in re.finditer(r"%([A-Za-z])", spec):
                    pattern += re.escape(spec[cursor : match.start()])
                    if match[1] not in _DATE_PARTS:
                        raise ValueError(f"Unsupported date directive: %{match[1]}")
                    pattern += _DATE_PARTS[match[1]][0]
                    cursor = match.end()
                pattern += re.escape(spec[cursor:])
            elif spec.endswith("d"):
                pattern = r"\d+"
            elif spec:
                raise ValueError(f"Unsupported template format: {spec!r}")
            else:
                pattern = r"[^/]+?"
            expression += f"(?P<{group}>{pattern})"
            self.fields.append((field, spec, group))
        self.regex = re.compile(expression)

    def parse(self, relative: str) -> dict[str, Any] | None:
        match = self.regex.fullmatch(relative)
        if match is None:
            return None
        values: dict[str, Any] = {}
        dates: dict[str, dict[str, int]] = {}
        for field, spec, group in self.fields:
            raw = match[group]
            if "%" in spec:
                # Validate even cycle labels without inventing a non-leap year.
                parsed = datetime.strptime(  # noqa: DTZ007 -- preserve source wall time
                    raw if "%Y" in spec else "2000" + raw,
                    spec if "%Y" in spec else "%Y" + spec,
                )
                parts = dates.setdefault(field, {})
                for directive in re.findall(r"%([A-Za-z])", spec):
                    key = _DATE_PARTS[directive][1]
                    value = getattr(parsed, key)
                    if key in parts and parts[key] != value:
                        raise ValueError(f"Conflicting {field} values in {relative}")
                    parts[key] = value
            else:
                value = int(raw) if spec.endswith("d") else raw
                if field in values and values[field] != value:
                    raise ValueError(f"Conflicting {field} values in {relative}")
                values[field] = value
        for field, parts in dates.items():
            if field == "cycle_time":
                values[field] = "".join(
                    f"{parts[key]:02d}" for key in ("month", "day", "hour")
                )
            else:
                values[field] = datetime(  # noqa: DTZ001 -- source wall time
                    **{"year": 1900, "month": 1, "day": 1, **parts}
                )
        return values

    def render(self, values: Mapping[str, Any]) -> str:
        values = dict(values)
        if isinstance(values.get("cycle_time"), str):
            values["cycle_time"] = datetime.strptime(  # noqa: DTZ007 -- cycle formatting only
                "2000" + values["cycle_time"], "%Y%m%d%H"
            )
        result = self.template.format_map(values)
        if Path(result).is_absolute() or ".." in Path(result).parts:
            raise ValueError("Rendered output path escapes the dataset root")
        return result


def templates(
    layout_config: Mapping[str, Any],
) -> list[tuple[str | None, PathTemplate]]:
    layout = layout_config["layout"]
    directory = layout.get("directory", "")
    names = layout.get("files", {None: layout.get("filename")})
    return [
        (group, PathTemplate("/".join(p for p in (directory, name) if p)))
        for group, name in names.items()
        if name is not None
    ]


def profile_layout(profile) -> dict:
    """Return independent IO rules, applying the selected forecast step."""
    layout = deepcopy(dict(profile.io_layout))
    if "hours_per_index" in layout.get("lead_time", {}):
        layout["lead_time"]["hours_per_index"] = profile.step_hours
    return layout
