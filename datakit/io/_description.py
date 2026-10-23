"""Load public dataset descriptions and parse their relative path templates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from string import Formatter
from typing import Any


def list_descriptions() -> tuple[str, ...]:
    return tuple(
        sorted(
            p.name[:-5]
            for p in files("datakit.io.descriptions").iterdir()
            if p.name.endswith(".yaml")
        )
    )


def load_description(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    import yaml

    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
    else:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            content = candidate.read_text(encoding="utf-8")
        else:
            name = str(value).removesuffix(".yaml")
            if name not in list_descriptions():
                raise ValueError(f"Unknown dataset description: {value!r}")
            content = (
                files("datakit.io.descriptions")
                .joinpath(name + ".yaml")
                .read_text(encoding="utf-8")
            )
        result = yaml.safe_load(content)
    if not isinstance(result, dict):
        raise TypeError("A dataset description must be a mapping")
    required = {"name", "implementation", "layout", "example"}
    if missing := required - result.keys():
        raise ValueError(f"Missing description fields: {sorted(missing)}")
    allowed = required | {
        "notes",
        "lead_time",
        "member",
        "structure",
        "cycle",
        "thresholds",
        "time_zone",
        "mapping",
    }
    if unknown := result.keys() - allowed:
        raise ValueError(f"Unknown description fields: {sorted(unknown)}")
    if "mapping" in result and not isinstance(result["mapping"], dict):
        raise TypeError("mapping must be a mapping")
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
        raise ValueError("The description example does not match its layout")
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


def templates(description: Mapping[str, Any]) -> list[tuple[str | None, PathTemplate]]:
    layout = description["layout"]
    directory = layout.get("directory", "")
    names = layout.get("files", {None: layout.get("filename")})
    return [
        (group, PathTemplate("/".join(p for p in (directory, name) if p)))
        for group, name in names.items()
        if name is not None
    ]
