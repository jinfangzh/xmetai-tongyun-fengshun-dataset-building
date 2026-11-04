"""Public file discovery records, independent of evaluation conventions."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileRecord:
    """A local file/store and the fields parsed from its path declaration."""

    path: Path
    fields: Mapping[str, Any]

    def __post_init__(self):
        if "://" in str(self.path):
            raise ValueError("FileRecord requires a local filesystem path")
        if not isinstance(self.fields, Mapping):
            raise TypeError("FileRecord fields must be a mapping")
        object.__setattr__(self, "path", Path(self.path).expanduser())
        object.__setattr__(self, "fields", dict(self.fields))
