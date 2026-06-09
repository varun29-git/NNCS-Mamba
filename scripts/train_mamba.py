"""Command-line wrapper for dual-head Mamba training."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nncs_mamba.training import main


if __name__ == "__main__":
    raise SystemExit(main())
