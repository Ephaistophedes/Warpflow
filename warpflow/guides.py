"""Surface-anchored, screen-sized stroke guides and their shared hit testing.

Projection and visibility are shared by drawing and picking: a hidden endpoint
must never capture a click through the mesh. The two saved anchors define a
straight guide, not a sampled path across a curved surface.
"""

import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector


HANDLE_RADIUS = 10.0
BODY_RADIUS = 6.0


def _region_3d(session):
    return session.area.spaces.active.region_3d


def _project(session, point):
    result = view3d_utils.location_3d_to_region_2d(
        session.region, _region_3d(session), Vector(point))
    if result is None or not np.all(np.isfinite(result)):
        return None
    return np.asarray(result, dtype=float)


def _visible(session, point, screen):
    """Test a world-space anchor against the session's immutable surface BVH."""
    if screen is None:
        return False
    region, rv3d = session.region, _region_3d(session)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, screen)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, screen)
    target_distance = (Vector(point) - origin).dot(direction)
    if target_distance < 0:
        return False
    # A small world-space allowance absorbs single-precision view matrix and
    # ray-triangle errors while retaining depth ordering on the far surface.
    tolerance = getattr(session, 'guide_tolerance', None)
    if tolerance is None:
        tolerance = max(float(np.linalg.norm(np.ptp(session.vertices, axis=0))) * 2e-5, 1e-6)
        session.guide_tolerance = tolerance
    hit, _, _, distance = session.bvh.ray_cast(origin, direction, target_distance + tolerance)
    return hit is None or distance >= target_distance - tolerance


def projected_strokes(session):
    """Yield visible anchors as (index, start_xy, tip_xy, start_ok, tip_ok)."""
    if not getattr(session.settings, 'show_strokes', True):
        return
    for index in range(len(session.obj.data.warpflow.strokes)):
        try:
            start, tip = session.stroke_endpoints(index)
            start_xy, tip_xy = _project(session, start), _project(session, tip)
            start_ok = _visible(session, start, start_xy)
            tip_ok = _visible(session, tip, tip_xy)
            if start_ok or tip_ok:
                yield index, start_xy, tip_xy, start_ok, tip_ok
        except (IndexError, ReferenceError, ValueError):
            # Saved indices can be invalid after topology editing outside a
            # paint session. Leave stale records for Clear Strokes to remove.
            continue
    # A newly mirrored counterpart has no saved record until release. Draw it
    # without adding it to hit testing or the native undo state.
    extra = getattr(session, 'mirror_preview_endpoints', lambda: None)()
    if extra is not None:
        start, tip = extra
        start_xy, tip_xy = _project(session, start), _project(session, tip)
        start_ok, tip_ok = _visible(session, start, start_xy), _visible(session, tip, tip_xy)
        if start_ok or tip_ok:
            yield -2, start_xy, tip_xy, start_ok, tip_ok


def _segment_distance(point, start, end):
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 < 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = np.clip(np.dot(point - start, delta) / length2, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + delta * fraction)))


def pick_stroke(session, mouse_region_xy):
    """Pick endpoint before shaft; the selected guide wins equal-distance ties."""
    mouse = np.asarray(mouse_region_xy, dtype=float)
    selected = getattr(session, 'selected_stroke', -1)
    handles, bodies = [], []
    for index, start, tip, start_ok, tip_ok in projected_strokes(session):
        if index < 0:
            continue
        priority = 0 if index == selected else 1
        for label, point, visible in (('START', start, start_ok), ('TIP', tip, tip_ok)):
            if visible:
                distance = float(np.linalg.norm(mouse - point))
                if distance <= HANDLE_RADIUS:
                    handles.append((distance, priority, index, label))
        if start_ok and tip_ok:
            distance = _segment_distance(mouse, start, tip)
            if distance <= BODY_RADIUS:
                bodies.append((distance, priority, index, 'BODY'))
    candidates = handles or bodies
    if not candidates:
        return None
    # Round subpixel differences so coincident guides reliably keep selection.
    winner = min(candidates, key=lambda item: (round(item[0], 2), item[1], -item[2]))
    return winner[2], winner[3]


def _handle(points, center, radius, square=False):
    if square:
        corners = center + np.array(((-radius, -radius), (radius, -radius),
                                     (radius, radius), (-radius, radius)))
    else:
        angles = np.arange(12) * (2.0 * np.pi / 12)
        corners = center + np.column_stack((np.cos(angles), np.sin(angles))) * radius
    for index, corner in enumerate(corners):
        points.extend((corner, corners[(index + 1) % len(corners)]))


def stroke_lines(session):
    """Return batched screen-space line vertices for normal and selected guides."""
    regular, selected = [], []
    selected_indices = {getattr(session, 'selected_stroke', -1)}
    if getattr(session.settings, 'mirror_x', False) and hasattr(session, '_partner') and session.selected_stroke >= 0:
        partner = session._partner(session.selected_stroke)
        if partner is not None:
            selected_indices.add(partner)
    for index, start, tip, start_ok, tip_ok in projected_strokes(session):
        is_selected = index in selected_indices
        points = selected if is_selected else regular
        if start_ok and tip_ok:
            delta = tip - start
            length = float(np.linalg.norm(delta))
            if length > 1:
                direction = delta / length
                side = np.array((-direction[1], direction[0]))
                back = tip - direction * min(14.0, length * 0.35)
                wing = min(6.0, length * 0.16)
                points.extend((start, tip, tip, back + side * wing, tip, back - side * wing))
        radius = 5.5 if is_selected else 4.0
        if start_ok:
            _handle(points, start, radius)
        if tip_ok:
            _handle(points, tip, radius, square=True)
    result = []
    for points in (regular, selected):
        array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        result.append(np.column_stack((array, np.zeros(len(array), dtype=np.float32))))
    return tuple(result)
