"""Use the host's SciPy, or our private Windows wheel matching Python's ABI.

No network calls, package installation, or modifications to Blender's runtime.
SciPy itself is BSD licensed; full wheel license notices accompany the binaries.
Unsupported hosts can still use the explicit graph approximation backend.
"""
import importlib.util
import platform
from pathlib import Path
import sys


def prepare():
    if importlib.util.find_spec("scipy") is not None:
        return
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        return
    abi = "cp%d%d" % sys.version_info[:2]
    vendor = Path(__file__).parent / "_vendor" / "win_amd64" / abi
    if (vendor / "scipy" / "__init__.py").is_file():
        # SciPy imports absolute scipy.* modules internally, requiring sys.path.
        # Append so a pre-existing host package always has priority.
        path = str(vendor)
        if path not in sys.path:
            sys.path.append(path)
