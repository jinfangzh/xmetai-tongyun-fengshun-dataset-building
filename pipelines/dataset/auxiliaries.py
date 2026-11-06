"""Prepare and copy declared dataset attachments; content checks belong to validation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from datakit.validation.zarr_inspection import file_sha256


@dataclass(frozen=True)
class PreparedAuxiliary:
    filename: str
    source: Path
    sha256: str


def prepare_auxiliaries(spec, inputs, destination, *, require_all=True):
    """Require explicit files for mandatory attachments; never create the store."""
    inputs = inputs or {}
    definitions = {item.filename: item for item in spec.variant.auxiliaries}
    unknown = sorted(inputs.keys() - definitions.keys())
    if unknown:
        raise ValueError(f"Auxiliaries are not declared in the target contract: {unknown}")
    missing = [name for name, item in definitions.items() if item.required and name not in inputs]
    if missing and require_all:
        raise ValueError(f"Required auxiliary inputs missing: {missing}; "
                         "provide a mapping from contract filenames to source paths")
    target = Path(destination).resolve()
    prepared = []
    for name, value in inputs.items():
        if Path(name).name != name:
            raise ValueError(f"Auxiliary filename must be a direct child: {name}")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Auxiliary {name}: source file does not exist: {path}")
        if path.is_relative_to(target):
            raise ValueError(f"Auxiliary source must be outside the new destination: {path}")
        prepared.append(PreparedAuxiliary(name, path, file_sha256(path)))
    return prepared


def copy_auxiliaries(prepared, destination):
    """Copy frozen assets without overwriting or modifying their original files."""
    root = Path(destination).resolve()
    for asset in prepared:
        if file_sha256(asset.source) != asset.sha256:
            raise ValueError(f"Auxiliary source changed before copying: {asset.source}")
        output = root / asset.filename
        if output.parent != root:
            raise ValueError(f"Auxiliary filename must be a direct child: {asset.filename}")
        with asset.source.open("rb") as source, output.open("xb") as target:
            shutil.copyfileobj(source, target)
        if file_sha256(output) != asset.sha256:
            raise ValueError(f"Auxiliary copy checksum mismatch: {output}")
