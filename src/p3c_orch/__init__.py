"""P3C-Orch reference implementation."""

from importlib.metadata import PackageNotFoundError, version

from .config import ProjectConfig, load_config
from .scheduler import P3CScheduler

__all__ = ["P3CScheduler", "ProjectConfig", "load_config"]
try:
    __version__ = version("p3c-orch")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
