"""Run numerical and actual Blender integration tests in isolated processes.

python scripts/test.py --blender "path/to/blender.exe"
Add --ui to run native simulated events and GPU drawing in a separate window.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    print('Running:', ' '.join(str(arg) for arg in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--ui', action='store_true')
    args = parser.parse_args()
    for name in ('test_geodesic.py', 'test_interpolation.py'):
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', name])
    for name in ('test_uv_bake_blender.py', 'test_proxy_blender.py', 'test_session_blender.py', 'test_dense_session_blender.py', 'test_stroke_edit_blender.py', 'test_mirror_blender.py', 'test_volumetric_blender.py'):
        run([args.blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python', 'tests/' + name])
    if args.ui:
        kwargs = {}
        if sys.platform == 'win32':
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startup
        for script, report in (('test_ui_blender.py', 'ui_result.json'), ('test_edit_ui_blender.py', 'edit_ui_result.json'), ('test_mirror_ui_blender.py', 'mirror_ui_result.json')):
            run([args.blender, '--factory-startup', '--enable-event-simulate',
                 '--window-geometry', '0', '0', '1360', '900', '--python', 'tests/' + script], **kwargs)
            result = json.loads((ROOT / 'tests' / report).read_text())
            if not result['passed']:
                raise SystemExit(result['error'])
    print('All selected Warpflow checks passed.')


if __name__ == '__main__':
    main()
