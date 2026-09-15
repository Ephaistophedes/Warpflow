"""Unpack the delivered ZIP in isolation and verify its actual Blender imports."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', default='blender')
    args = parser.parse_args()
    archive_path = ROOT / 'dist' / 'Warpflow-1.1.0-windows-x64.zip'
    # Context cleanup runs after the Blender subprocess releases native DLLs.
    with tempfile.TemporaryDirectory(prefix='warpflow-package-test-') as folder:
        isolated = Path(folder).resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if not (isolated / member).resolve().is_relative_to(isolated):
                    raise ValueError('Archive path escapes the isolated test directory')
            archive.extractall(isolated)
        script = isolated / 'verify_package.py'
        script.write_text('''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bpy
import numpy as np
import warpflow
warpflow.register()
from warpflow.geodesic import create_solver
import scipy
assert str(Path(__file__).parent) in scipy.__file__, scipy.__file__
solver = create_solver(np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]), np.array([[0,1,2],[0,2,3]]))
assert solver.stats['factorization_count'] == 2, solver.backend
distance = solver.distances([0])
assert np.all(np.isfinite(distance)) and distance[2] > distance[1]
assert 'Warpflow' in warpflow.bl_info['name']
assert warpflow.bl_info['version'] == (1, 1, 0)
from warpflow.properties import WARPFLOW_PG_stroke
assert 'tip_vertices' in WARPFLOW_PG_stroke.bl_rna.properties.keys()
warpflow.unregister()
warpflow.register()
warpflow.unregister()
print('DELIVERED_ZIP_OK', bpy.app.version_string, scipy.__version__, scipy.__file__)
''')
        subprocess.run([args.blender, '--background', '--factory-startup', '--python-exit-code', '1',
                        '--python', str(script)], check=True)


if __name__ == '__main__':
    main()
