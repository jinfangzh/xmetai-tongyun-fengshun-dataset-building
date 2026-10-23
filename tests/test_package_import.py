import subprocess
import sys
from pathlib import Path


def test_clean_source_import_without_residual_directories(tmp_path):
    package = Path(__file__).resolve().parents[1] / "datakit"
    for source in package.rglob("*.py"):
        target = tmp_path / "datakit" / source.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import datakit; "
            "assert all(hasattr(datakit, name) for name in datakit.__all__)",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
