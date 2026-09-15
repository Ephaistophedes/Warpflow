"""Native simulated UI regression for retained arrows, endpoint edits and delete.

Run in a separate Blender UI process with --factory-startup --enable-event-simulate.
This uses Blender's event queue only, never the operating-system mouse/keyboard.
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
checks = []
started = time.perf_counter()
window = area = region = None
last_pointer = (0, 0)
NAME = 'Warpflow Editable Stroke Test'


def count():
    return len(bpy.data.objects[NAME].data.warpflow.strokes)


def field():
    return session.read_field(bpy.data.objects[NAME].data).copy()


def ends(index=0):
    return np.array(session.ACTIVE.stroke_endpoints(index))


def pointer(point, value=None):
    global last_pointer
    pixel = location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(point))
    assert pixel is not None
    x, y = int(region.x + pixel.x), int(region.y + pixel.y)
    last_pointer = (x, y)
    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    if value:
        window.event_simulate(type='LEFTMOUSE', value=value, x=x, y=y)


def key(kind, **modifiers):
    # Key events use the same in-viewport cursor position as the latest pointer.
    for value in ('PRESS', 'RELEASE'):
        window.event_simulate(type=kind, value=value, x=last_pointer[0], y=last_pointer[1], **modifiers)


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
        bpy.context.object.name = NAME
        rv = area.spaces.active.region_3d
        rv.view_perspective = 'ORTHO'
        rv.view_rotation = Quaternion((1, 0, 0, 0))
        rv.view_distance = 7
        rv.view_location = Vector((0, 0, 0))
        area.spaces.active.overlay.show_floor = False
        area.spaces.active.overlay.show_axis_x = False
        area.spaces.active.overlay.show_axis_y = False
        bpy.ops.warpflow.enter()
    yield
    first = np.array([[-1., -.6, 0], [-.3, -.6, 0]])
    second = np.array([[.7, .2, 0], [.7, 1.05, 0]])
    for stroke_index, (start, tip) in enumerate((first, second)):
        pointer(start, 'PRESS')
        yield
        pointer(tip)
        yield
        if stroke_index == 0:
            pointer([2.25, -1.5, 0])
            yield
            pointer([2.25, -1.5, 0], 'RELEASE')
        else:
            pointer(tip, 'RELEASE')
        yield
    assert count() == 2
    for index, expected in enumerate((first, second)):
        np.testing.assert_allclose(ends(index), expected, atol=.025)
        assert bpy.data.objects[NAME].data.warpflow.strokes[index].has_tip
    checks.append('Completed mouse strokes preserve both surface endpoints')
    checks.append('Releasing off the surface retains the last valid tip and direction')
    original_ends = ends()
    original_field = field()
    tip_target = np.array([-.45, .05, 0])
    pointer(original_ends[1], 'PRESS')
    yield
    assert session.ACTIVE.edit is not None and session.ACTIVE.edit['endpoint'] == 'TIP'
    assert session.ACTIVE.selected_stroke == 0
    pointer(tip_target)
    yield
    assert not np.allclose(field(), original_field)
    assert count() == 2
    pointer(tip_target, 'RELEASE')
    yield
    assert session.ACTIVE.edit is None and count() == 2
    np.testing.assert_allclose(ends()[1], tip_target, atol=.025)
    np.testing.assert_allclose(ends()[0], original_ends[0], atol=1e-6)
    checks.append('Tip handle is picked and dragged; field updates before release')
    changed_ends = ends()
    key('Z', ctrl=True)
    yield
    np.testing.assert_allclose(ends(), original_ends, atol=1e-6)
    key('Z', ctrl=True, shift=True)
    yield
    np.testing.assert_allclose(ends(), changed_ends, atol=1e-6)
    checks.append('Native undo and redo each restore one endpoint edit')
    start_target = np.array([-1.4, -.95, 0])
    pointer(ends()[0], 'PRESS')
    yield
    assert session.ACTIVE.edit is not None and session.ACTIVE.edit['endpoint'] == 'START'
    pointer(start_target)
    yield
    pointer(start_target, 'RELEASE')
    yield
    np.testing.assert_allclose(ends()[0], start_target, atol=.025)
    np.testing.assert_allclose(ends()[1], changed_ends[1], atol=1e-6)
    assert count() == 2
    checks.append('Start handle moves the source while keeping the tip attached')
    saved_ends, saved_field = ends(), field()
    pointer(saved_ends[1], 'PRESS')
    yield
    pointer([-.7, .4, 0])
    yield
    key('ESC')
    yield
    assert session.ACTIVE is not None and session.ACTIVE.edit is None
    np.testing.assert_allclose(ends(), saved_ends, atol=1e-6)
    np.testing.assert_array_equal(field(), saved_field)
    checks.append('Esc cancels an endpoint edit without leaving paint mode')
    pointer(np.mean(saved_ends, axis=0), 'PRESS')
    yield
    pointer(np.mean(saved_ends, axis=0), 'RELEASE')
    yield
    assert session.ACTIVE.selected_stroke == 0 and session.ACTIVE.edit is None
    key('DEL')
    yield
    assert count() == 1
    np.testing.assert_allclose(field(), np.tile([0, 1], (len(field()), 1)), atol=.025)
    key('Z', ctrl=True)
    yield
    assert count() == 2
    np.testing.assert_allclose(ends(), saved_ends, atol=1e-6)
    np.testing.assert_allclose(field(), saved_field, atol=1e-6)
    checks.append('Clicking an arrow body selects it; Delete removes it and undo restores it')
    # Deleting with no selection must never fall through to object deletion.
    session.ACTIVE.selected_stroke = -1
    key('X')
    yield
    assert session.ACTIVE is not None and count() == 2
    checks.append('Delete/X without a stroke selection leaves the mesh intact')
    # Select for the visual artifact and show the Warpflow sidebar.
    pointer(np.mean(saved_ends, axis=0), 'PRESS')
    pointer(np.mean(saved_ends, axis=0), 'RELEASE')
    for kind, value in (('MOUSEMOVE', 'NOTHING'), ('LEFTMOUSE', 'PRESS'), ('LEFTMOUSE', 'RELEASE')):
        window.event_simulate(type=kind, value=value,
                              x=area.x + area.width - 12, y=window.height - 325)
    yield
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=2)
        bpy.ops.screen.screenshot(filepath=str(ROOT / 'tests' / 'stroke_edit_viewport.png'))
    # Return cursor to viewport before Escape and capture the outside-mode guides.
    pointer([0, -1.5, 0])
    key('ESC')
    yield
    assert session.ACTIVE is None and count() == 2
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=2)
        bpy.ops.screen.screenshot(filepath=str(ROOT / 'tests' / 'stroke_guides_after_exit.png'))
    # Undo and screenshot operators can change the timer callback's context
    # cache; give the entry operator its current object explicitly.
    obj = bpy.data.objects[NAME]
    with bpy.context.temp_override(window=window, area=area, region=region, object=obj, active_object=obj):
        bpy.ops.warpflow.enter()
    yield
    np.testing.assert_allclose(ends(), saved_ends, atol=1e-6)
    checks.append('Retained arrows draw after exit and endpoints survive re-entry')
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.warpflow.exit()
    warpflow.unregister()
    checks.append('Clean exit and unregister after editing')


scenario_steps = scenario()


def tick():
    error = None
    try:
        next(scenario_steps)
        return .22
    except StopIteration:
        pass
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    (ROOT / 'tests' / 'edit_ui_result.json').write_text(json.dumps({
        'blender': bpy.app.version_string, 'passed': error is None,
        'checks': checks, 'error': error, 'seconds': time.perf_counter() - started,
    }, indent=2))
    session.end_session()
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(tick, first_interval=1.0)
