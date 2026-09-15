"""Session precompute and transactional stroke state, independent of event dispatch.

Heat factorization and UV/BVH/proxy setup occur in Enter, never modal.invoke.
Sources are stroke START points: the distance solve runs once on mouse-down;
only the direction and O(V) weighted blend change during dragging. Extending a
stroke to path samples is possible by adding sources here, but is outside v1.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import time
import uuid

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .geodesic import create_solver
from .interpolation import FieldAccumulator
from .uv import build_uv_data

ATTRIBUTE = "Warpflow"
ACTIVE = None
CACHE_BYTES = 128 * 1024 * 1024


def mesh_arrays(mesh):
    mesh.calc_loop_triangles()
    vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', vertices)
    triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', triangles)
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get('vertices', edges)
    return vertices.reshape(-1, 3), triangles.reshape(-1, 3), edges.reshape(-1, 2)


def geometry_signature(mesh, matrix):
    vertices, triangles, edges = mesh_arrays(mesh)
    digest = hashlib.blake2b(digest_size=20)
    for array in (vertices, triangles, edges, np.array(matrix, dtype=np.float64)):
        digest.update(array.tobytes())
    if mesh.uv_layers.active:
        uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        mesh.uv_layers.active.uv.foreach_get('vector', uv)
        digest.update(uv.tobytes())
    return digest.hexdigest()


def ensure_attribute(mesh):
    attr = mesh.color_attributes.get(ATTRIBUTE)
    if attr and (attr.domain != 'POINT' or attr.data_type != 'FLOAT_COLOR'):
        raise ValueError('Existing Warpflow attribute must be Point / Float Color; rename it before painting.')
    if not attr:
        attr = mesh.color_attributes.new(name=ATTRIBUTE, type='FLOAT_COLOR', domain='POINT')
        write_field(mesh, np.zeros((len(mesh.vertices), 2)))
    mesh.color_attributes.active_color = attr
    return attr


def read_field(mesh):
    attr = mesh.color_attributes.get(ATTRIBUTE)
    if attr is None:
        return np.zeros((len(mesh.vertices), 2))
    values = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
    attr.data.foreach_get('color', values)
    return values.reshape(-1, 4)[:, :2].astype(np.float64) * 2.0 - 1.0


def write_field(mesh, field):
    attr = mesh.color_attributes.get(ATTRIBUTE)
    if attr is None:
        raise ValueError('Warpflow color attribute is missing.')
    rgba = np.empty((len(field), 4), dtype=np.float32)
    rgba[:, :2] = field * 0.5 + 0.5
    rgba[:, 2] = 0.0
    rgba[:, 3] = 1.0
    attr.data.foreach_set('color', rgba.ravel())
    mesh.update()


def stroke_endpoints(obj, stroke, uv=None):
    """Return a saved arrow's world endpoints without changing its metadata.

    Version 1.0 stored only a source and UV direction. Its arrows get an inferred
    length until edited; simply displaying an older file must not rewrite it.
    """
    mesh = obj.data
    start = sum((mesh.vertices[int(vertex)].co * float(weight)
                 for vertex, weight in zip(stroke.vertices, stroke.barycentric)), Vector())
    start_world = obj.matrix_world @ start
    if getattr(stroke, 'has_tip', False):
        tip = sum((mesh.vertices[int(vertex)].co * float(weight)
                   for vertex, weight in zip(stroke.tip_vertices, stroke.tip_barycentric)), Vector())
        return start_world, obj.matrix_world @ tip
    uv = uv if uv is not None else build_uv_data(mesh)
    ids = tuple(int(value) for value in stroke.vertices)
    triangle = next((tri.index for tri in mesh.loop_triangles
                     if tuple(tri.vertices) == ids), None)
    if triangle is None:
        raise ValueError('A saved stroke no longer matches the mesh triangles.')
    direction = (uv.triangle_tangents[triangle] * float(stroke.direction[0])
                 + uv.triangle_bitangents[triangle] * float(stroke.direction[1]))
    world_direction = obj.matrix_world.to_3x3() @ Vector(direction)
    world_direction.normalize()
    bounds = np.asarray([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    length = max(float(np.linalg.norm(np.ptp(bounds, axis=0))) * 0.08, 1e-6)
    return start_world, start_world + world_direction * length


def validate_object(obj):
    if obj is None or obj.type != 'MESH' or obj.mode != 'OBJECT':
        raise ValueError('Select one mesh in Object Mode.')
    if obj.library or obj.data.library or not obj.is_editable:
        raise ValueError('Make the mesh local and editable before painting.')
    if obj.data.users > 1:
        raise ValueError('Make the mesh single-user before painting (Object > Relations > Make Single User).')
    if not obj.data.polygons:
        raise ValueError('The mesh needs faces and an existing UV unwrap.')
    if any(mod.show_viewport for mod in obj.modifiers):
        raise ValueError('Disable viewport modifiers before painting the base mesh; re-enable them after painting.')
    if abs(obj.matrix_world.determinant()) < 1e-12:
        raise ValueError('Object scale must be nonzero.')
    if not obj.data.uv_layers.active:
        raise ValueError('UV unwrap the mesh before entering Flow Paint Mode.')


class PaintSession:
    def __init__(self, context, progress=None):
        obj = context.object
        validate_object(obj)
        self.object_name = obj.name
        self.matrix = obj.matrix_world.copy()
        self.inverse_matrix = self.matrix.inverted()
        self.signature = geometry_signature(obj.data, self.matrix)
        if obj.data.warpflow.strokes and obj.data.warpflow.geometry_signature != self.signature:
            raise ValueError('Geometry, transform or UVs changed since painting. Clear Strokes before starting a new field.')
        self.area = context.area
        self.window = context.window
        self.region = next((r for r in self.area.regions if r.type == 'WINDOW'), None) if self.area else None
        self.settings = context.scene.warpflow
        self.scene_name = context.scene.name
        self.settings_dirty = False
        self.ended = False
        self.drag = None
        self.edit = None
        self.selected_stroke = -1
        self._legacy_endpoints = {}
        self.draw = None
        self.timer = None
        self.preview_saved = None
        self.distance_cache = OrderedDict()
        self.cache_bytes = 0
        self.proxy = None
        self.proxy_solver = None
        self.proxy_accumulator = None
        self.notice = ''
        self.last_preview_ms = 0.0
        start = time.perf_counter()
        update = progress or (lambda value: None)
        update(0.02)
        self.local_vertices, self.triangles, self.edges = mesh_arrays(obj.data)
        normals = np.empty(len(self.local_vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get('normal', normals)
        self.local_normals = normals.reshape(-1, 3)
        m = np.asarray(self.matrix, dtype=np.float64)
        self.vertices = self.local_vertices @ m[:3, :3].T + m[:3, 3]
        self.bvh = BVHTree.FromPolygons(self.vertices.tolist(), self.triangles.tolist(), all_triangles=True)
        # Reflection is in object-local space, independent of object placement.
        self.mirror_bvh = BVHTree.FromPolygons(self.local_vertices.tolist(), self.triangles.tolist(), all_triangles=True)
        self.mirror_tolerance = max(float(np.linalg.norm(np.ptp(self.local_vertices, axis=0))) * 0.02, 1e-6)
        self.uv = build_uv_data(obj.data)
        update(0.12)
        self.solver = create_solver(self.vertices, self.triangles, self.edges, progress=lambda v: update(0.12 + 0.48 * v))
        # Benchmark a solve at entry. A slow solve also selects proxy preview,
        # even when the configurable vertex threshold was not reached.
        start_solve = time.perf_counter()
        self.solver.distances([int(self.triangles[0, 0])])
        self.solve_ms = (time.perf_counter() - start_solve) * 1000
        needs_proxy = len(self.vertices) > self.settings.proxy_threshold or self.solve_ms > 33
        face_edges = np.concatenate((self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]))
        face_edges.sort(axis=1)
        edge_pairs = np.sort(self.edges, axis=1)
        stride = len(self.vertices)
        has_wire_edges = np.any(~np.isin(edge_pairs[:, 0].astype(np.int64) * stride + edge_pairs[:, 1],
                                        face_edges[:, 0].astype(np.int64) * stride + face_edges[:, 1]))
        if needs_proxy and has_wire_edges:
            # Decimating faces cannot preserve loose-edge endpoints in general.
            # Keep the full graph so thin wire bridges do not disappear during
            # preview. This is an explicit topology-safe full-resolution fallback.
            self.notice = 'Wire edges require full-resolution preview to preserve connectivity.'
        elif needs_proxy:
            from .proxy import build_proxy
            self.proxy = build_proxy(obj, self.vertices, self.triangles, self.solver.components,
                                     target_vertices=min(self.settings.proxy_target, max(200, len(self.vertices) // 2)))
            self.notice = '; '.join(reason for _, reason in self.proxy.stats.get('fallbacks', []))
            if self.proxy.is_proxy:
                self.proxy_solver = create_solver(self.proxy.vertices, self.proxy.triangles)
            else:
                self.proxy = None
        update(0.85)
        ensure_attribute(obj.data)
        obj.data.warpflow.geometry_signature = self.signature
        self.rebuild()
        update(1.0)
        self.setup_seconds = time.perf_counter() - start

    @property
    def obj(self):
        return bpy.data.objects.get(self.object_name)

    def rebind(self):
        obj = self.obj
        scene = bpy.data.scenes.get(self.scene_name)
        if obj is None or scene is None or obj.type != 'MESH':
            return False
        self.settings = scene.warpflow
        return geometry_signature(obj.data, obj.matrix_world) == self.signature

    def _distance(self, vertices, weights, proxy=False):
        # LRU keeps memory O(V * cached_sources), with a 128 MiB upper budget.
        # Cache misses after undo/re-entry reuse factorization, never rebuild it.
        key = (proxy, tuple(int(v) for v in vertices), tuple(round(float(v), 7) for v in weights))
        if key in self.distance_cache:
            self.distance_cache.move_to_end(key)
            return self.distance_cache[key]
        if proxy:
            vertices, weights = self.proxy.map_source(vertices, weights)
            result = self.proxy_solver.distances(vertices, weights)
        else:
            result = self.solver.distances(vertices, weights)
        while self.distance_cache and self.cache_bytes + result.nbytes > CACHE_BYTES:
            _, old = self.distance_cache.popitem(last=False)
            self.cache_bytes -= old.nbytes
        if result.nbytes <= CACHE_BYTES:
            self.distance_cache[key] = result
            self.cache_bytes += result.nbytes
        return result

    def _accumulator(self, solver):
        return FieldAccumulator(len(solver.components), solver.components,
                                sharpness=self.settings.sharpness,
                                distance_scale=solver.mean_edge_length)

    def rebuild(self):
        self.accumulator = self._accumulator(self.solver)
        self.proxy_accumulator = self._accumulator(self.proxy_solver) if self.proxy_solver else None
        for stroke in self.obj.data.warpflow.strokes:
            self.accumulator.append(self._distance(stroke.vertices, stroke.barycentric), stroke.direction, stroke.strength)
            if self.proxy_accumulator is not None:
                self.proxy_accumulator.append(self._distance(stroke.vertices, stroke.barycentric, True), stroke.direction, stroke.strength)
        write_field(self.obj.data, self.accumulator.field)
        self.settings_dirty = False
        if self.draw:
            self.draw.update_arrows(self.accumulator.field)

    def barycentric_source(self, triangle, position):
        """Convert a world-space surface hit to a stable mesh anchor."""
        if triangle is None or not 0 <= int(triangle) < len(self.triangles):
            raise ValueError('The stroke endpoint needs a mesh triangle.')
        ids = self.triangles[int(triangle)]
        tri = self.vertices[ids]
        basis = np.stack((tri[1] - tri[0], tri[2] - tri[0]), axis=1)
        coords = np.linalg.lstsq(basis, np.asarray(position) - tri[0], rcond=None)[0]
        bary = np.clip([1.0 - coords.sum(), coords[0], coords[1]], 0.0, 1.0)
        if not np.all(np.isfinite(bary)) or np.sum(bary) <= 0:
            raise ValueError('The stroke endpoint is not a finite surface point.')
        bary /= np.sum(bary)
        return ids.copy(), bary

    def _source_position(self, vertices, barycentric):
        return Vector(np.asarray(barycentric) @ self.vertices[np.asarray(vertices, dtype=np.int32)])

    def _source_triangle(self, vertices):
        matches = np.flatnonzero(np.all(self.triangles == np.asarray(vertices), axis=1))
        if not len(matches):
            raise ValueError('A saved stroke no longer matches the mesh triangles.')
        return int(matches[0])

    @staticmethod
    def _record_data(stroke):
        return {'vertices': tuple(stroke.vertices), 'barycentric': tuple(stroke.barycentric),
                'direction': tuple(stroke.direction), 'strength': float(stroke.strength),
                'mirror_pair': str(getattr(stroke, 'mirror_pair', '')),
                'has_tip': bool(getattr(stroke, 'has_tip', False)),
                'tip_vertices': tuple(getattr(stroke, 'tip_vertices', (0, 0, 0))),
                'tip_barycentric': tuple(getattr(stroke, 'tip_barycentric', (0.0, 0.0, 0.0)))}

    @staticmethod
    def _write_record(stroke, data):
        # Convert NumPy scalars explicitly for Blender RNA vector properties.
        stroke.vertices = tuple(int(v) for v in data['vertices'])
        stroke.barycentric = tuple(float(v) for v in data['barycentric'])
        stroke.direction = tuple(float(v) for v in data['direction'])
        stroke.strength = float(data['strength'])
        stroke.mirror_pair = data.get('mirror_pair', '')
        stroke.has_tip = bool(data.get('has_tip', False))
        if stroke.has_tip:
            stroke.tip_vertices = tuple(int(v) for v in data['tip_vertices'])
            stroke.tip_barycentric = tuple(float(v) for v in data['tip_barycentric'])

    def stroke_endpoints(self, index):
        """Return the completed arrow, or its current uncommitted edit."""
        if self.edit is not None and self.edit['index'] == index:
            return self.edit['position'].copy(), self.edit['tip_position'].copy()
        if self.edit is not None and self.edit.get('mirror_index') == index and self.edit.get('mirror_record') is not None:
            return self._data_endpoints(self.edit['mirror_record'])
        stroke = self.obj.data.warpflow.strokes[index]
        if getattr(stroke, 'has_tip', False):
            return (self._source_position(stroke.vertices, stroke.barycentric),
                    self._source_position(stroke.tip_vertices, stroke.tip_barycentric))
        key = (tuple(stroke.vertices), tuple(stroke.barycentric), tuple(stroke.direction))
        if key not in self._legacy_endpoints:
            self._legacy_endpoints[key] = stroke_endpoints(self.obj, stroke, self.uv)
        start, tip = self._legacy_endpoints[key]
        return start.copy(), tip.copy()

    def _data_endpoints(self, data):
        return (self._source_position(data['vertices'], data['barycentric']),
                self._source_position(data['tip_vertices'], data['tip_barycentric']))

    def _partner(self, index):
        records = self.obj.data.warpflow.strokes
        pair = records[index].mirror_pair
        if pair:
            return next((j for j, record in enumerate(records) if j != index and record.mirror_pair == pair), None)
        return None

    def mirrored_record(self, data):
        """Reflect both local surface anchors, then encode in the OTHER UV frame.

        Nearest-surface projection permits modest asymmetry (2% mesh diagonal).
        Side checks prevent a one-sided mesh from mirroring back onto itself.
        A mirrored component may be disconnected from the source component.
        """
        if not data.get('has_tip', False):
            return None
        anchors = []
        for world in self._data_endpoints(data):
            local = self.inverse_matrix @ world
            target = Vector((-local.x, local.y, local.z))
            hit, _, triangle, distance = self.mirror_bvh.find_nearest(target)
            if hit is None or distance > self.mirror_tolerance:
                return None
            if abs(target.x) > self.mirror_tolerance * .001 and hit.x * target.x <= 0:
                return None
            ids, bary = self.barycentric_source(triangle, self.matrix @ hit)
            anchors.append((ids, bary, triangle, hit))
        start, tip = anchors
        if self.solver.components[start[0][0]] != self.solver.components[tip[0][0]]:
            return None
        direction = self.uv.direction_to_uv(start[2], tip[3] - start[3])
        if direction is None:
            return None
        return {'vertices': start[0], 'barycentric': start[1], 'tip_vertices': tip[0],
                'tip_barycentric': tip[1], 'has_tip': True, 'direction': direction,
                'strength': data['strength'], 'mirror_pair': data.get('mirror_pair', '')}

    def _coincident_mirror(self, original, mirrored):
        if mirrored is None:
            return False
        tolerance = max(float(np.linalg.norm(np.ptp(self.vertices, axis=0))) * 1e-6, 1e-8)
        return np.allclose(self._data_endpoints(original), self._data_endpoints(mirrored), rtol=0, atol=tolerance)

    def mirror_preview_endpoints(self):
        """Additional transient arrow for the GPU overlay, with no RNA changes."""
        if self.drag and self.drag.get('mirror_x') and self.drag.get('direction') is not None:
            mirrored = self.mirrored_record(self.drag)
            if mirrored is not None and not self._coincident_mirror(self.drag, mirrored):
                return self._data_endpoints(mirrored)
        if self.edit and self.edit.get('mirror_x') and self.edit.get('mirror_index') is None:
            mirrored = self.edit.get('mirror_record')
            if mirrored is not None and not self._coincident_mirror(self.edit, mirrored):
                return self._data_endpoints(mirrored)
        return None

    def begin_stroke(self, triangle, position, mouse):
        self.cancel_edit()
        self.cancel_stroke()
        ids, bary = self.barycentric_source(triangle, position)
        self.selected_stroke = -1
        self.drag = {'triangle': triangle, 'vertices': ids.copy(), 'barycentric': bary,
                     'position': self._source_position(ids, bary), 'mouse_start': tuple(mouse),
                     'mouse_end': tuple(mouse), 'direction': None, 'pending': False,
                     'strength': self.settings.strength, 'before': read_field(self.obj.data),
                     'has_tip': False, 'mirror_x': self.settings.mirror_x}
        self.drag['distance'] = self._distance(ids, bary, self.proxy is not None)

    def set_stroke_tip(self, triangle, position):
        """Retain the actual surface tip of a new gesture when available."""
        if self.drag is None:
            return False
        if triangle is None:
            self.drag['has_tip'] = False
            return False
        ids, bary = self.barycentric_source(triangle, position)
        if self.solver.components[ids[0]] != self.solver.components[self.drag['vertices'][0]]:
            return False
        self.drag.update(tip_vertices=ids, tip_barycentric=bary,
                         tip_position=self._source_position(ids, bary), has_tip=True)
        return True

    def preview(self):
        if not self.drag or self.drag['direction'] is None:
            return
        start = time.perf_counter()
        accum = self.proxy_accumulator if self.proxy is not None else self.accumulator
        mirrored = self.mirrored_record(self.drag) if self.drag.get('mirror_x') else None
        if mirrored is not None and not self._coincident_mirror(self.drag, mirrored):
            candidate = deepcopy(accum)
            candidate.append(self.drag['distance'], self.drag['direction'], self.drag['strength'])
            candidate.append(self._distance(mirrored['vertices'], mirrored['barycentric'], self.proxy is not None),
                             mirrored['direction'], mirrored['strength'])
            field = candidate.field
        else:
            field = accum.preview(self.drag['distance'], self.drag['direction'], self.drag['strength'])
        if self.proxy is not None:
            field = self.proxy.upsample(field)
        write_field(self.obj.data, field)
        if self.draw:
            self.draw.update_arrows(field)
        self.drag['pending'] = False
        self.last_preview_ms = (time.perf_counter() - start) * 1000

    def commit_stroke(self):
        drag = self.drag
        if drag is None:
            return False
        if drag['direction'] is None or drag['strength'] <= 0:
            self.cancel_stroke()
            return False
        # Full-fidelity solve is required on release even in proxy preview mode.
        distances = self._distance(drag['vertices'], drag['barycentric'])
        records = self.obj.data.warpflow.strokes
        original_count = len(records)
        mirrored = self.mirrored_record(drag) if drag.get('mirror_x') else None
        if self._coincident_mirror(drag, mirrored):
            mirrored = None
        if mirrored is not None:
            drag['mirror_pair'] = mirrored['mirror_pair'] = uuid.uuid4().hex
        elif drag.get('mirror_x') and self.mirrored_record(drag) is None:
            self.notice = 'Mirror X: no matching opposite surface; kept the original stroke.'
        try:
            stroke = records.add()
            self._write_record(stroke, drag)
            self.accumulator.append(distances, drag['direction'], drag['strength'])
            if self.proxy_accumulator is not None:
                self.proxy_accumulator.append(drag['distance'], drag['direction'], drag['strength'])
            if mirrored is not None:
                self._write_record(records.add(), mirrored)
                self.accumulator.append(self._distance(mirrored['vertices'], mirrored['barycentric']),
                                        mirrored['direction'], mirrored['strength'])
                if self.proxy_accumulator is not None:
                    self.proxy_accumulator.append(self._distance(mirrored['vertices'], mirrored['barycentric'], True),
                                                  mirrored['direction'], mirrored['strength'])
            write_field(self.obj.data, self.accumulator.field)
        except Exception:
            while len(records) > original_count:
                records.remove(len(records) - 1)
            self.rebuild()
            raise
        self.drag = None
        self.selected_stroke = original_count
        if self.draw:
            self.draw.update_arrows(self.accumulator.field)
        return True

    def cancel_stroke(self):
        if self.drag is not None and self.obj:
            write_field(self.obj.data, self.drag['before'])
            self.drag = None
            if self.draw:
                self.draw.update_arrows(read_field(self.obj.data))

    def begin_edit(self, index, endpoint, mouse):
        """Prepare replay caches once; endpoint motion changes only a draft."""
        if endpoint not in {'START', 'TIP'}:
            return False
        records = self.obj.data.warpflow.strokes
        if not 0 <= index < len(records):
            return False
        self.cancel_stroke()
        self.cancel_edit()
        original = self._record_data(records[index])
        start, tip = self.stroke_endpoints(index)
        draft = dict(original)
        triangle = self._source_triangle(draft['vertices'])
        component = self.solver.components[draft['vertices'][0]]
        if not draft['has_tip']:
            # Old files have no actual tip. Anchor their inferred handle to the
            # same surface when editing; a click/cancel leaves RNA untouched.
            hit, _, tip_triangle, _ = self.bvh.find_nearest(tip)
            if hit is None or self.solver.components[self.triangles[tip_triangle, 0]] != component:
                hit, tip_triangle = tip, triangle
            tip_ids, tip_bary = self.barycentric_source(tip_triangle, hit)
            draft.update(tip_vertices=tip_ids, tip_barycentric=tip_bary, has_tip=True)
            tip = self._source_position(tip_ids, tip_bary)
        use_proxy = self.proxy is not None
        prefix = self._accumulator(self.proxy_solver if use_proxy else self.solver)
        for record in records[:index]:
            prefix.append(self._distance(record.vertices, record.barycentric, use_proxy),
                          record.direction, record.strength)
        # Keep cached suffix distances alive throughout this gesture. Their
        # source points never move, even when an older stroke's tail is edited.
        suffix = [(self._distance(record.vertices, record.barycentric, use_proxy),
                   tuple(record.direction), float(record.strength)) for record in records[index + 1:]]
        draft.update(index=index, endpoint=endpoint, triangle=triangle, component=component,
                     position=start, tip_position=tip, mouse_start=tuple(mouse),
                     mouse_end=tuple(mouse), pending=False, changed=False,
                     before=read_field(self.obj.data), original=original, prefix=prefix,
                     suffix=suffix, distance=self._distance(draft['vertices'], draft['barycentric'], use_proxy))
        draft['mirror_x'] = self.settings.mirror_x
        draft['mirror_index'] = self._partner(index) if draft['mirror_x'] else None
        if draft['mirror_x']:
            draft['mirror_record'] = self.mirrored_record(draft)
            first = min(index, draft['mirror_index']) if draft['mirror_index'] is not None else index
            prefix = self._accumulator(self.proxy_solver if use_proxy else self.solver)
            for record in records[:first]:
                prefix.append(self._distance(record.vertices, record.barycentric, use_proxy), record.direction, record.strength)
            draft['prefix'] = prefix
            # Either half may be selected: retain history order even when the
            # counterpart occurs before the selected stroke.
            draft['mirror_tail'] = [(j, self._record_data(records[j]),
                                     self._distance(records[j].vertices, records[j].barycentric, use_proxy))
                                    for j in range(first, len(records))]
        self.edit = draft
        self.selected_stroke = index
        return True

    def update_edit(self, triangle, position, mouse):
        """Update surface anchors and UV direction; defer distance work to TIMER."""
        edit = self.edit
        if edit is None or triangle is None or not 0 <= int(triangle) < len(self.triangles):
            return False
        if self.solver.components[self.triangles[int(triangle), 0]] != edit['component']:
            return False
        ids, bary = self.barycentric_source(triangle, position)
        hit = self._source_position(ids, bary)
        start = hit if edit['endpoint'] == 'START' else edit['position']
        tip = hit if edit['endpoint'] == 'TIP' else edit['tip_position']
        source_triangle = int(triangle) if edit['endpoint'] == 'START' else edit['triangle']
        direction = self.uv.direction_to_uv(source_triangle, self.inverse_matrix.to_3x3() @ (tip - start))
        if direction is None:
            return False
        candidate = dict(edit, direction=direction)
        if edit['endpoint'] == 'START':
            candidate.update(vertices=ids, barycentric=bary)
        else:
            candidate.update(tip_vertices=ids, tip_barycentric=bary, has_tip=True)
        mirrored = self.mirrored_record(candidate) if edit.get('mirror_x') else None
        if edit.get('mirror_x') and mirrored is None:
            self.notice = 'Mirror X: move the handle where an opposite surface exists.'
            return False
        if edit['endpoint'] == 'START':
            edit.update(vertices=ids, barycentric=bary, triangle=source_triangle, position=start)
        else:
            edit.update(tip_vertices=ids, tip_barycentric=bary, has_tip=True, tip_position=tip)
        edit.update(direction=direction, mouse_end=tuple(mouse), pending=True)
        if edit.get('mirror_x'):
            edit['mirror_record'] = mirrored
        original = edit['original']
        edit['changed'] = (not np.array_equal(edit['vertices'], original['vertices'])
                           or not np.allclose(edit['barycentric'], original['barycentric'], rtol=0, atol=1e-7)
                           or not np.allclose(direction, original['direction'], rtol=0, atol=1e-7)
                           or not original['has_tip']
                           or not np.array_equal(edit['tip_vertices'], original['tip_vertices'])
                           or not np.allclose(edit['tip_barycentric'], original['tip_barycentric'], rtol=0, atol=1e-7))
        return True

    def preview_edit(self):
        edit = self.edit
        if edit is None or not edit['pending']:
            return
        start = time.perf_counter()
        accum = deepcopy(edit['prefix'])
        if edit['endpoint'] == 'START':
            edit['distance'] = self._distance(edit['vertices'], edit['barycentric'], self.proxy is not None)
        if edit.get('mirror_x'):
            mirrored = edit.get('mirror_record')
            duplicate = self._coincident_mirror(edit, mirrored)
            for index, data, distance in edit['mirror_tail']:
                if index == edit['index']:
                    accum.append(edit['distance'], edit['direction'], edit['strength'])
                elif index == edit['mirror_index']:
                    if not duplicate and mirrored is not None:
                        accum.append(self._distance(mirrored['vertices'], mirrored['barycentric'], self.proxy is not None),
                                     mirrored['direction'], mirrored['strength'])
                else:
                    accum.append(distance, data['direction'], data['strength'])
            if edit['mirror_index'] is None and mirrored is not None and not duplicate:
                accum.append(self._distance(mirrored['vertices'], mirrored['barycentric'], self.proxy is not None),
                             mirrored['direction'], mirrored['strength'])
        else:
            accum.append(edit['distance'], edit['direction'], edit['strength'])
            for distances, direction, strength in edit['suffix']:
                accum.append(distances, direction, strength)
        field = self.proxy.upsample(accum.field) if self.proxy is not None else accum.field
        write_field(self.obj.data, field)
        if self.draw:
            self.draw.update_arrows(field)
        edit['pending'] = False
        self.last_preview_ms = (time.perf_counter() - start) * 1000

    def commit_edit(self):
        edit = self.edit
        if edit is None:
            return False
        if not edit['changed']:
            self.cancel_edit()
            return False
        records = self.obj.data.warpflow.strokes
        original_records = [self._record_data(record) for record in records]
        stroke = records[edit['index']]
        try:
            if edit.get('mirror_x'):
                mirrored = edit.get('mirror_record')
                if mirrored is not None and not self._coincident_mirror(edit, mirrored):
                    pair = edit.get('mirror_pair') or uuid.uuid4().hex
                    edit['mirror_pair'] = mirrored['mirror_pair'] = pair
                    target = records[edit['mirror_index']] if edit['mirror_index'] is not None else records.add()
                    self._write_record(target, mirrored)
                else:
                    edit['mirror_pair'] = ''
            else:
                # Mirror OFF means edits affect only the selected arrow. Unlink
                # its former partner so toggling ON cannot overwrite stale links.
                pair = edit.get('mirror_pair')
                for record in records:
                    if pair and record.mirror_pair == pair:
                        record.mirror_pair = ''
                edit['mirror_pair'] = ''
            self._write_record(stroke, edit)
            if edit.get('mirror_x') and self._coincident_mirror(edit, edit.get('mirror_record')) and edit['mirror_index'] is not None:
                records.remove(edit['mirror_index'])
                self.selected_stroke = edit['index'] - int(edit['mirror_index'] < edit['index'])
            # A moved source always commits using the full-resolution solver.
            self.rebuild()
        except Exception:
            records.clear()
            for original in original_records:
                self._write_record(records.add(), original)
            self.edit = None
            self.rebuild()
            raise
        self.edit = None
        return True

    def cancel_edit(self):
        if self.edit is not None:
            edit, self.edit = self.edit, None
            if self.obj:
                write_field(self.obj.data, edit['before'])
                if self.draw:
                    self.draw.update_arrows(edit['before'])

    def remove_selected(self):
        self.cancel_stroke()
        self.cancel_edit()
        records = self.obj.data.warpflow.strokes
        index = self.selected_stroke
        if not 0 <= index < len(records):
            self.selected_stroke = -1
            return False
        original = [self._record_data(stroke) for stroke in records]
        partner = self._partner(index)
        try:
            indices = [index, partner] if self.settings.mirror_x and partner is not None else [index]
            if partner is not None and not self.settings.mirror_x:
                records[partner].mirror_pair = ''
            for remove_index in sorted(indices, reverse=True):
                records.remove(remove_index)
            self.rebuild()
        except Exception:
            records.clear()
            for data in original:
                self._write_record(records.add(), data)
            self.rebuild()
            raise
        self.selected_stroke = -1
        return True

    def update_preview_shading(self):
        if not self.area or self.area.type != 'VIEW_3D':
            return
        space = self.area.spaces.active
        shading = space.shading
        if self.settings.show_preview and self.preview_saved is None:
            self.preview_saved = (shading.type, shading.color_type, shading.light)
            # SpaceView3D survives in area.spaces when another editor is active.
            # Keep its reference so changing editors cannot strand our shading.
            self.preview_space = space
            shading.type, shading.color_type, shading.light = 'SOLID', 'VERTEX', 'FLAT'
        elif not self.settings.show_preview:
            self.restore_shading()

    def restore_shading(self):
        if self.preview_saved is not None:
            try:
                space = getattr(self, 'preview_space', None)
                if space is not None:
                    shading = space.shading
                    shading.type, shading.color_type, shading.light = self.preview_saved
            except (ReferenceError, RuntimeError):
                # A closed area no longer has a viewport whose state can persist.
                pass
            finally:
                self.preview_saved = None
                self.preview_space = None

    def close(self, context=None):
        if self.ended:
            return
        self.ended = True
        try:
            self.cancel_stroke()
            self.cancel_edit()
        except (ReferenceError, RuntimeError, ValueError):
            self.drag = None
            self.edit = None
        if self.draw:
            self.draw.remove()
            self.draw = None
        if self.timer:
            try:
                bpy.context.window_manager.event_timer_remove(self.timer)
            except (ReferenceError, RuntimeError):
                pass
            self.timer = None
        try:
            self.restore_shading()
            if self.window:
                self.window.cursor_modal_restore()
            if self.area:
                self.area.header_text_set(None)
                self.area.tag_redraw()
        except (ReferenceError, RuntimeError):
            pass


def end_session(context=None):
    global ACTIVE
    if ACTIVE is not None:
        ACTIVE.close(context)
        ACTIVE = None


@bpy.app.handlers.persistent
def undo_redo_post(_):
    if ACTIVE is None:
        return
    try:
        # Blender may recreate every RNA pointer on undo. Retain only immutable
        # NumPy caches and locate the object/settings again by saved name.
        ACTIVE.drag = None
        ACTIVE.edit = None
        ACTIVE.selected_stroke = -1
        ACTIVE._legacy_endpoints.clear()
        if not ACTIVE.rebind():
            end_session()
            return
        ensure_attribute(ACTIVE.obj.data)
        ACTIVE.rebuild()
        # Rebuild clears settings_dirty, so restored preview settings need their
        # own synchronization rather than waiting for a later property callback.
        ACTIVE.update_preview_shading()
        if ACTIVE.area:
            ACTIVE.area.tag_redraw()
    except Exception:
        end_session()


@bpy.app.handlers.persistent
def load_pre(_):
    end_session()
