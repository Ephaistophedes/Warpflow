"""Actual modal events for Mirror X creation, live guide, editing and pair undo."""
import json
from pathlib import Path
import sys
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
checks = []
last = (0, 0)


def pointer(point, value=None, **mods):
    global last
    p = location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(point))
    last = (int(region.x + p.x), int(region.y + p.y))
    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=last[0], y=last[1], **mods)
    if value:
        window.event_simulate(type='LEFTMOUSE', value=value, x=last[0], y=last[1], **mods)


def key(kind, **mods):
    for value in ('PRESS', 'RELEASE'):
        window.event_simulate(type=kind, value=value, x=last[0], y=last[1], **mods)


def records():
    return bpy.data.objects['Mirror UI'].data.warpflow.strokes


def scenario():
    global window, area, region
    warpflow.register()
    window = bpy.context.window
    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    area.spaces.active.show_region_ui = True
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=25, y_subdivisions=25, size=4)
        bpy.context.object.name = 'Mirror UI'
        rv = area.spaces.active.region_3d
        rv.view_perspective = 'ORTHO'
        rv.view_rotation = Quaternion((1, 0, 0, 0))
        rv.view_distance = 7
        rv.view_location = Vector((0, 0, 0))
        bpy.context.scene.warpflow.mirror_x = True
        bpy.ops.warpflow.enter()
    yield
    pointer((.7, -.6, 0), 'PRESS')
    yield
    pointer((1.3, .5, 0))
    yield
    assert len(records()) == 0
    assert session.ACTIVE.mirror_preview_endpoints() is not None
    checks.append('Mirror ON previews a reflected arrow and field before commit')
    pointer((1.3, .5, 0), 'RELEASE')
    yield
    assert len(records()) == 2 and records()[0].mirror_pair == records()[1].mirror_pair
    key('Z', ctrl=True)
    yield
    assert len(records()) == 0
    key('Z', ctrl=True, shift=True)
    yield
    assert len(records()) == 2
    checks.append('One native undo/redo step covers both generated strokes')
    pointer(session.ACTIVE.stroke_endpoints(1)[0], 'PRESS')
    yield
    assert session.ACTIVE.edit is not None and session.ACTIVE.edit['index'] == 1
    pointer((-.4, -1.0, 0))
    yield
    pointer((-.4, -1.0, 0), 'RELEASE')
    yield
    a, b = (np.array(session.ACTIVE.stroke_endpoints(j)) for j in (0, 1))
    np.testing.assert_allclose(a * [-1, 1, 1], b, atol=2e-6)
    checks.append('Editing the negative-X copy updates its positive-X partner')
    for kind, value in (('MOUSEMOVE', 'NOTHING'), ('LEFTMOUSE', 'PRESS'), ('LEFTMOUSE', 'RELEASE')):
        window.event_simulate(type=kind, value=value, x=area.x + area.width - 12, y=window.height - 325)
    yield
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=2)
        bpy.ops.screen.screenshot(filepath=str(ROOT / 'tests' / 'mirror_viewport.png'))
    pointer((-.4, -1., 0))
    key('DEL')
    yield
    assert len(records()) == 0
    key('Z', ctrl=True)
    yield
    assert len(records()) == 2
    checks.append('Delete removes the selected pair; undo restores both')
    bpy.context.scene.warpflow.mirror_x = False
    pointer((.2, .6, 0), 'PRESS', ctrl=True)
    yield
    pointer((.6, 1.1, 0), ctrl=True)
    yield
    pointer((.6, 1.1, 0), 'RELEASE', ctrl=True)
    yield
    assert len(records()) == 3
    checks.append('Mirror OFF creates one stroke and leaves existing pairs intact')
    key('ESC')
    yield
    assert session.ACTIVE is None
    warpflow.unregister()


steps = scenario()


def tick():
    error = None
    try:
        next(steps)
        return .25
    except StopIteration:
        pass
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    (ROOT / 'tests' / 'mirror_ui_result.json').write_text(json.dumps({
        'blender': bpy.app.version_string, 'passed': error is None, 'checks': checks, 'error': error}, indent=2))
    session.end_session()
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(tick, first_interval=1)
