"""Configuration model for the isolated legacy Zarr refresh pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
@dataclass(frozen=True)
class RefreshProfile:
    key: str
    input: Path
    output_name: str
    cadence_hours: int
    radiation_seconds: int | None = None
    channel_coordinate: str = "auto"
    statistics: str = "auto"
    statistics_divide: dict[str, float] = field(default_factory=dict)
    statistics_multiply: dict[str, float] = field(default_factory=dict)
    channel_rename: dict[str, str] = field(default_factory=dict)
    mask: dict[str, Any] = field(default_factory=dict)
    expected_channels: int | None = None
    expected_grid: tuple[int, int] | None = None
    already_refreshed: str = "error"
    notes: str = ""

    @classmethod
    def from_dict(cls, key: str, value: dict[str, Any]) -> "RefreshProfile":
        grid = value.get("expected_grid")
        return cls(
            key=key,
            input=Path(value["input"]),
            output_name=str(value["output_name"]),
            cadence_hours=int(value["cadence_hours"]),
            radiation_seconds=(
                None
                if value.get("radiation_seconds") is None
                else int(value["radiation_seconds"])
            ),
            channel_coordinate=str(value.get("channel_coordinate", "auto")),
            statistics=str(value.get("statistics", "auto")),
            statistics_divide={
                str(name): float(divisor)
                for name, divisor in value.get("statistics_divide", {}).items()
            },
            statistics_multiply={
                str(pattern): float(factor)
                for pattern, factor in value.get("statistics_multiply", {}).items()
            },
            channel_rename={
                str(source): str(target)
                for source, target in value.get("channel_rename", {}).items()
            },
            mask=dict(value.get("mask", {})),
            expected_channels=(
                None
                if value.get("expected_channels") is None
                else int(value["expected_channels"])
            ),
            expected_grid=None if grid is None else (int(grid[0]), int(grid[1])),
            already_refreshed=str(value.get("already_refreshed", "error")),
            notes=str(value.get("notes", "")),
        )

    def serializable(self) -> dict[str, Any]:
        result = {
            "key": self.key,
            "input": str(self.input),
            "output_name": self.output_name,
            "cadence_hours": self.cadence_hours,
            "radiation_seconds": self.radiation_seconds,
            "channel_coordinate": self.channel_coordinate,
            "statistics": self.statistics,
            "statistics_divide": self.statistics_divide,
            "statistics_multiply": self.statistics_multiply,
            "channel_rename": self.channel_rename,
            "mask": self.mask,
            "expected_channels": self.expected_channels,
            "expected_grid": self.expected_grid,
            "already_refreshed": self.already_refreshed,
            "notes": self.notes,
        }
        return result


def load_profiles(path: Path) -> dict[str, RefreshProfile]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("datasets")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a non-empty 'datasets' object")
    return {key: RefreshProfile.from_dict(key, value) for key, value in raw.items()}


def select_profiles(
    profiles: dict[str, RefreshProfile], requested: list[str]
) -> list[RefreshProfile]:
    if not requested or requested == ["all"]:
        return list(profiles.values())
    missing = [name for name in requested if name not in profiles]
    if missing:
        raise KeyError("unknown dataset profile(s): " + ", ".join(missing))
    return [profiles[name] for name in requested]
