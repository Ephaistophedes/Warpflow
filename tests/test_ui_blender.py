"""Real-window GPU + modal event integration; run a separate factory-startup UI.

blender --factory-startup --enable-event-simulate --python tests/test_ui_blender.py
The timer queues Blender's native test events; no OS mouse/keyboard is touched.
Outputs tests/ui_result.json and tests/viewport.png and closes its own process.
"""
import json
from pathlib import Path
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Quaternion, Vector
from bpy_extras.view3d_utils import location_3d_to_region_2d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warpflow
from warpflow import session
bpy.context.preferences.view.show_splash = False

stage = 0
checks = []
started = time.perf_counter()
window = None
area = None
region = None
points = None
obj_name = 'Warpflow UI Test'


def result(error=None):
    (ROOT / 'tests' / 'ui_result.json').write_text(json.dumps({
        'blender': bpy.app.version_string, 'passed': error is None,
        'checks': checks, 'error': error, 'seconds': time.perf_counter() - started,
    }, indent=2))


def event(kind, value='PRESS', point=0, **modifiers):
    p = points[point]
    window.event_simulate(type=kind, value=value, x=p[0], y=p[1], **modifiers)


def field():
    return session.read_field(bpy.data.objects[obj_name].data)


def count():
    return len(bpy.data.objects[obj_name].data.warpflow.strokes)


def run():
    global stage, window, area, region, points
    try:
        if stage == 0:
            warpflow.register()
            window = bpy.context.window
            area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
            area.spaces.active.show_region_ui = True
            region = next(r for r in area.regions if r.type == 'WINDOW')
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete(use_global=False)
                bpy.ops.mesh.primitive_grid_add(x_subdivisions=25, y_subdivisions=25, size=4)
                obj = bpy.context.object
                obj.name = obj_name
                assert obj.data.uv_layers.active
                rv3d = area.spaces.active.region_3d
                rv3d.view_perspective = 'ORTHO'
                rv3d.view_rotation = Quaternion((1, 0, 0, 0))
                rv3d.view_distance = 7
                rv3d.view_location = Vector((0, 0, 0))
                area.spaces.active.overlay.show_floor = False
                area.spaces.active.overlay.show_axis_x = False
                area.spaces.active.overlay.show_axis_y = False
                bpy.ops.warpflow.enter()
                assert session.ACTIVE is not None
                assert len(session.ACTIVE.draw.handlers) == 2
                checks.append('Enter operator, precompute, GPU shader and two draw handlers')
        elif stage == 1:
            # Select the visible Warpflow sidebar tab in this fixed test window.
            # active_panel_category is read-only in Blender 5.x.
            for kind, value in (('MOUSEMOVE', 'NOTHING'), ('LEFTMOUSE', 'PRESS'), ('LEFTMOUSE', 'RELEASE')):
                window.event_simulate(type=kind, value=value,
                                      x=area.x + area.width - 12, y=window.height - 325)
            rv3d = area.spaces.active.region_3d
            points = []
            for point in [(-1, -.7, 0), (-.3, -.7, 0), (1, .5, 0), (1, 1.2, 0)]:
                p = location_3d_to_region_2d(region, rv3d, Vector(point))
                assert p is not None
                points.append((int(p.x + region.x), int(p.y + region.y)))
            event('MOUSEMOVE', 'NOTHING', 0)
            event('LEFTMOUSE', 'PRESS', 0)
        elif stage == 2:
            assert session.ACTIVE.drag is not None, 'Native LMB did not begin stroke'
            event('MOUSEMOVE', 'NOTHING', 1)
        elif stage == 3:
            assert count() == 0
            np.testing.assert_allclose(field(), np.tile([1, 0], (len(field()), 1)), atol=.02)
            checks.append('Native live drag updates full field before commit')
            event('LEFTMOUSE', 'RELEASE', 1)
        elif stage == 4:
            assert count() == 1
            assert session.ACTIVE.drag is None
            event('LEFTMOUSE', 'PRESS', 2)
        elif stage == 5:
            event('MOUSEMOVE', 'NOTHING', 3)
        elif stage == 6:
            assert count() == 1
            event('LEFTMOUSE', 'RELEASE', 3)
        elif stage == 7:
            assert count() == 2
            np.testing.assert_allclose(np.linalg.norm(field(), axis=1), 1, atol=2e-6)
            assert np.ptp(field()[:, 0]) > .2, 'Two-source field should vary over mesh'
            checks.append('Two committed strokes produce normalized varying field')
            event('Z', 'PRESS', 0, ctrl=True)
        elif stage == 8:
            assert count() == 1, f'Undo left {count()} strokes'
            np.testing.assert_allclose(field(), np.tile([1, 0], (len(field()), 1)), atol=.02)
            checks.append('Native Ctrl Z removes exactly one completed stroke')
            event('Z', 'RELEASE', 0, ctrl=True)
            event('Z', 'PRESS', 0, ctrl=True, shift=True)
        elif stage == 9:
            assert count() == 2
            checks.append('Native Shift Ctrl Z restores stroke and field')
            event('Z', 'RELEASE', 0, ctrl=True, shift=True)
            event('LEFTMOUSE', 'PRESS', 0)
        elif stage == 10:
            event('MOUSEMOVE', 'NOTHING', 3)
        elif stage == 11:
            event('RIGHTMOUSE', 'PRESS', 3)
        elif stage == 12:
            assert count() == 2 and session.ACTIVE.drag is None
            checks.append('Right mouse cancels preview without adding undo/constraint')
            # The screenshot forces actual viewport draw callback execution.
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=2)
                bpy.ops.screen.screenshot(filepath=str(ROOT / 'tests' / 'viewport.png'))
            event('ESC', 'PRESS', 0)
        elif stage == 13:
            assert session.ACTIVE is None and count() == 2
            checks.append('Esc preserves committed field and removes session resources')
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.context.scene.warpflow.resolution = '512'
                output = ROOT / 'tests' / 'ui_flowmap.png'
                out = bpy.ops.warpflow.export(filepath=str(output))
                assert out == {'FINISHED'} and output.is_file()
            checks.append('Export operator produces native baked PNG after exit')
            warpflow.unregister()
            checks.append('Clean unregister after modal painting')
            result()
            bpy.ops.wm.quit_blender()
            return None
        stage += 1
        return .4
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        result(error)
        session.end_session()
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(run, first_interval=1.0)
