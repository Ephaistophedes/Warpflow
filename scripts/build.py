"""Build the legacy installable ZIP without modifying Blender's installation."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'warpflow'
OUTPUT = ROOT / 'dist' / 'Warpflow-1.1.0-windows-x64.zip'


def build():
    for abi in ('cp311', 'cp313'):
        if not (PACKAGE / '_vendor' / 'win_amd64' / abi / '.wheel-sha256').is_file():
            raise SystemExit('Run python scripts/vendor_scipy.py to fetch the pinned wheels first.')
    OUTPUT.parent.mkdir(exist_ok=True)
    sources = [(p, 'warpflow/' + p.relative_to(PACKAGE).as_posix())
               for p in PACKAGE.rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}]
    sources += [(ROOT / name, 'warpflow/' + name) for name in ('README.md', 'LICENSE', 'THIRD_PARTY.md')]
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in sorted(sources, key=lambda pair: pair[1]):
            # Stable metadata makes builds reproducible from identical inputs.
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert 'warpflow/__init__.py' in archive.namelist()
        assert not any('__pycache__' in name for name in archive.namelist())
    checksum = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix('.zip.sha256').write_text(checksum + '  ' + OUTPUT.name + '\n')
    print(json.dumps({'zip': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
                      'files': len(sources), 'sha256': checksum}, indent=2))


if __name__ == '__main__':
    build()
