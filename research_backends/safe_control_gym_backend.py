"""
Optional Safe-Control-Gym backend boundary.

This module is intentionally small: it does not reimplement quadrotor dynamics
or MPC. Its job is to make the trusted-backend dependency explicit and provide
a single place where the research plant/controller adapter will be added once
Safe-Control-Gym is installed and pinned.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Optional


TRUSTED_BACKEND_NAME = "Safe-Control-Gym"
TRUSTED_BACKEND_PACKAGE = "safe_control_gym"
TRUSTED_BACKEND_SOURCE = "https://github.com/utiasDSL/safe-control-gym"
RESEARCH_BACKEND_PROVENANCE = "established-benchmark"


@dataclass(frozen=True)
class SafeControlGymBackendStatus:
    """Availability and provenance for the optional Safe-Control-Gym backend."""

    available: bool
    package: str
    source: str
    provenance: str
    import_path: Optional[str] = None


def get_safe_control_gym_status() -> SafeControlGymBackendStatus:
    """Return whether Safe-Control-Gym is importable in this Python environment."""
    spec = find_spec(TRUSTED_BACKEND_PACKAGE)
    return SafeControlGymBackendStatus(
        available=spec is not None,
        package=TRUSTED_BACKEND_PACKAGE,
        source=TRUSTED_BACKEND_SOURCE,
        provenance=RESEARCH_BACKEND_PROVENANCE,
        import_path=spec.origin if spec is not None else None,
    )


def require_safe_control_gym() -> SafeControlGymBackendStatus:
    """
    Require Safe-Control-Gym before constructing research plant/MPC adapters.

    Raises:
        RuntimeError: If Safe-Control-Gym is not installed.
    """
    status = get_safe_control_gym_status()
    if not status.available:
        raise RuntimeError(
            "Safe-Control-Gym is not installed. Install and pin the benchmark "
            "before using the research-grade quadrotor plant or MPC expert. "
            f"Source: {status.source}"
        )
    return status
