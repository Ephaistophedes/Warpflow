"""Batched GPU overlays for the field and editable, persistent stroke guides.

Arrow construction is O(min(V, maximum_arrows)); every-N sampling keeps density
bounded independently of subdivision. Native attribute shading draws the mesh.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .guides import stroke_lines


def _draw_batch(shader, batch, color, width, depth='NONE'):
    if batch is None:
        return
    old_blend = gpu.state.blend_get()
    old_depth = gpu.state.depth_test_get()
    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set(depth)
        shader.bind()
        shader.uniform_float('viewportSize', gpu.state.viewport_get()[2:])
        shader.uniform_float('lineWidth', width)
        shader.uniform_float('color', color)
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set(old_depth)
        gpu.state.blend_set(old_blend)


def _draw_guides(shader, state):
    normal, selected = stroke_lines(state)
    all_points = np.concatenate((normal, selected))
    if not len(all_points):
        return
    # One outline batch plus two color batches, independent of stroke count.
    outline = batch_for_shader(shader, 'LINES', {'pos': all_points})
    _draw_batch(shader, outline, (0.015, 0.025, 0.035, 0.8), 5.0)
    if len(normal):
        batch = batch_for_shader(shader, 'LINES', {'pos': normal})
        _draw_batch(shader, batch, (0.78, 1.0, 0.94, 1.0), 2.5)
    if len(selected):
        batch = batch_for_shader(shader, 'LINES', {'pos': selected})
        _draw_batch(shader, batch, (1.0, 0.55, 0.08, 1.0), 3.0)


_guide_handler = None
_guide_shader = None
_saved_cache = None


@bpy.app.handlers.persistent
def _invalidate_saved_guides(*_):
    global _saved_cache
    _saved_cache = None


@bpy.app.handlers.persistent
def _saved_geometry_changed(_scene, depsgraph):
    if _saved_cache is None:
        return
    for update in depsgraph.updates:
        if update.is_updated_geometry and update.id.original.as_pointer() in _saved_cache['ids']:
            _invalidate_saved_guides()
            return


def _saved_state(context):
    """Read saved anchors without solver setup; cache geometry across redraws."""
    from types import SimpleNamespace
    from mathutils.bvhtree import BVHTree
    from .session import mesh_arrays, stroke_endpoints
    from .uv import build_uv_data

    global _saved_cache
    obj = context.active_object
    mesh = obj.data
    records = mesh.warpflow.strokes
    snapshot = tuple((tuple(r.vertices), tuple(r.barycentric), tuple(r.direction),
                      bool(r.has_tip), tuple(r.tip_vertices), tuple(r.tip_barycentric)) for r in records)
    key = (obj.as_pointer(), mesh.as_pointer(), tuple(v for row in obj.matrix_world for v in row),
           len(mesh.vertices), len(mesh.polygons), snapshot)
    if _saved_cache is None or _saved_cache['key'] != key:
        local, triangles, _ = mesh_arrays(mesh)
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        vertices = local @ matrix[:3, :3].T + matrix[:3, 3]
        bvh = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
        uv = build_uv_data(mesh) if any(not r.has_tip for r in records) else None
        endpoints = []
        for record in records:
            try:
                endpoints.append(stroke_endpoints(obj, record, uv=uv))
            except (IndexError, ValueError):
                endpoints.append(None)
        _saved_cache = {'key': key, 'ids': (obj.as_pointer(), mesh.as_pointer()),
                        'vertices': vertices, 'bvh': bvh, 'endpoints': endpoints,
                        'tolerance': max(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 2e-5, 1e-6)}

    endpoints = _saved_cache['endpoints']

    def get_endpoints(index):
        result = endpoints[index]
        if result is None:
            raise ValueError('Stroke refers to removed mesh vertices.')
        return result

    return SimpleNamespace(obj=obj, region=context.region, area=context.area,
                           settings=context.scene.warpflow, selected_stroke=-1,
                           vertices=_saved_cache['vertices'], bvh=_saved_cache['bvh'],
                           guide_tolerance=_saved_cache['tolerance'], stroke_endpoints=get_endpoints)


def _draw_saved_guides():
    from . import session
    global _guide_shader
    context = bpy.context
    if session.ACTIVE is not None:
        return
    obj = context.active_object
    if (context.area is None or context.area.type != 'VIEW_3D' or context.region.type != 'WINDOW'
            or obj is None or obj.type != 'MESH' or obj.mode != 'OBJECT'
            or not hasattr(obj.data, 'warpflow') or not obj.data.warpflow.strokes
            or not context.scene.warpflow.show_strokes
            or not context.space_data.overlay.show_overlays or not obj.visible_get()):
        return
    if context.space_data.local_view and not obj.local_view_get(context.space_data):
        return
    try:
        state = _saved_state(context)
        if _guide_shader is None:
            _guide_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        _draw_guides(_guide_shader, state)
    except (ReferenceError, RuntimeError, ValueError, IndexError):
        # Topology/UV editing can invalidate old guides; painting itself gives
        # the actionable geometry-signature error if the user re-enters.
        _invalidate_saved_guides()


def register_guides():
    global _guide_handler
    if _guide_handler is None:
        _guide_handler = bpy.types.SpaceView3D.draw_handler_add(_draw_saved_guides, (), 'WINDOW', 'POST_PIXEL')
    for handlers, callback in ((bpy.app.handlers.depsgraph_update_post, _saved_geometry_changed),
                               (bpy.app.handlers.undo_post, _invalidate_saved_guides),
                               (bpy.app.handlers.redo_post, _invalidate_saved_guides),
                               (bpy.app.handlers.load_pre, _invalidate_saved_guides)):
        if callback not in handlers:
            handlers.append(callback)


def unregister_guides():
    global _guide_handler, _guide_shader
    if _guide_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_guide_handler, 'WINDOW')
        _guide_handler = None
    for handlers, callback in ((bpy.app.handlers.depsgraph_update_post, _saved_geometry_changed),
                               (bpy.app.handlers.undo_post, _invalidate_saved_guides),
                               (bpy.app.handlers.redo_post, _invalidate_saved_guides),
                               (bpy.app.handlers.load_pre, _invalidate_saved_guides)):
        if callback in handlers:
            handlers.remove(callback)
    _guide_shader = None
    _invalidate_saved_guides()


class FlowDrawing:
    def __init__(self, session):
        self.session = session
        self.shader = None
        self.arrow_batch = None
        self.handlers = []

    def install(self):
        self.shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        self.handlers.append(bpy.types.SpaceView3D.draw_handler_add(self.draw_arrows, (), 'WINDOW', 'POST_VIEW'))
        self.handlers.append(bpy.types.SpaceView3D.draw_handler_add(self.draw_stroke, (), 'WINDOW', 'POST_PIXEL'))
        self.update_arrows(self.session.accumulator.field)

    def remove(self):
        for handler in self.handlers:
            bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
        self.handlers.clear()
        self.arrow_batch = None
        self.shader = None

    def update_arrows(self, field):
        if self.shader is None:
            return
        s = self.session
        step = max(1, int(np.ceil(len(field) / s.settings.arrow_count)))
        ids = np.arange(0, len(field), step)
        ids = ids[np.linalg.norm(field[ids], axis=1) > 0.01]
        if len(ids) == 0:
            self.arrow_batch = None
            return
        basis_u, basis_v = s.uv.vertex_tangents[ids], s.uv.vertex_bitangents[ids]
        direction = basis_u * field[ids, :1] + basis_v * field[ids, 1:2]
        matrix = np.asarray(s.matrix, dtype=np.float64)[:3, :3]
        direction = direction @ matrix.T
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        # UV handedness flips on mirrored islands; geometry normals do not.
        normal = s.local_normals[ids] @ np.linalg.inv(matrix)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        side = np.cross(normal, direction)
        diagonal = np.linalg.norm(np.ptp(s.vertices, axis=0))
        length = max(diagonal * 0.012, 1e-6) * s.settings.arrow_scale
        start = s.vertices[ids] + normal * length * 0.06
        tip = start + direction * length
        back = tip - direction * length * 0.3
        points = np.stack((start, tip, tip, back + side * length * 0.16,
                           tip, back - side * length * 0.16), axis=1).reshape(-1, 3)
        self.arrow_batch = batch_for_shader(self.shader, 'LINES', {'pos': points.astype(np.float32)})

    def _visible(self):
        return (not self.session.ended and bpy.context.area == self.session.area
                and bpy.context.region == self.session.region)

    def _draw(self, batch, color, width, depth):
        _draw_batch(self.shader, batch, color, width, depth)

    def draw_arrows(self):
        if self._visible() and self.session.settings.show_arrows:
            self._draw(self.arrow_batch, (0.06, 0.85, 1.0, 0.95), 1.5, 'LESS_EQUAL')

    def draw_stroke(self):
        if not self._visible():
            return
        _draw_guides(self.shader, self.session)
        if not self.session.settings.show_stroke or not self.session.drag:
            return
        drag = self.session.drag
        start = np.array(drag['mouse_start'], dtype=float)
        end = np.array(drag['mouse_end'], dtype=float)
        delta = end - start
        norm = np.linalg.norm(delta)
        if norm < 1:
            return
        direction = delta / norm
        side = np.array([-direction[1], direction[0]])
        back = end - direction * min(14.0, norm * 0.3)
        points = np.stack((start, end, end, back + side * 5, end, back - side * 5))
        points = np.column_stack((points, np.zeros(6)))
        batch = batch_for_shader(self.shader, 'LINES', {'pos': points.astype(np.float32)})
        self._draw(batch, (1.0, 0.72, 0.15, 1.0), 3.0, 'NONE')
