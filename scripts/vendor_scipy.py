"""Reproducibly vendor official SciPy wheels; never invokes pip or changes Blender.

Run with system Python. Hashes are pinned to PyPI release metadata. Wheel licenses
and dist-info are preserved. Native modules must be extracted to load on Windows.
"""
import hashlib
from pathlib import Path
import urllib.request
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
WHEELS = {
    "cp311": (
        "https://files.pythonhosted.org/packages/d6/73/c449a7d56ba6e6f874183759f8483cde21f900a8be117d67ffbb670c2958/scipy-1.16.2-cp311-cp311-win_amd64.whl",
        "91e9e8a37befa5a69e9cacbe0bcb79ae5afb4a0b130fd6db6ee6cc0d491695fa",
    ),
    "cp313": (
        "https://files.pythonhosted.org/packages/a1/57/0f38e396ad19e41b4c5db66130167eef8ee620a49bc7d0512e3bb67e0cab/scipy-1.16.2-cp313-cp313-win_amd64.whl",
        "fda714cf45ba43c9d3bae8f2585c777f64e3f89a2e073b668b32ede412d8f52c",
    ),
}

for abi, (url, expected) in WHEELS.items():
    dest = ROOT / "warpflow" / "_vendor" / "win_amd64" / abi
    marker = dest / ".wheel-sha256"
    if marker.exists() and marker.read_text() == expected:
        print(abi, "already verified")
        continue
    data = urllib.request.urlopen(url, timeout=90).read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(dest)
    marker.write_text(expected)
    print(abi, "vendored", len(data), "bytes")
