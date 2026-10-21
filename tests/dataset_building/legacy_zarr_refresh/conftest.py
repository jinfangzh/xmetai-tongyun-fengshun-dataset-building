"""Import the self-contained refresh example without importing parent examples."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE_PARENT = Path(__file__).resolve().parents[3] / "examples" / "dataset_building"
if str(EXAMPLE_PARENT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_PARENT))
