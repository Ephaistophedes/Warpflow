"""Versioned guide files and transactional imports onto a matching base mesh."""

from collections import Counter
import json
import math
import os
import tempfile
import uuid

import bpy
import numpy as np
from mathutils import Matrix

from . import session


FORMAT = 'warpflow-guides'
VERSION = 1


def _mesh_info(mesh):
    # Guides use local triangle anchors and UV directions. Object placement is
    # deliberately excluded so a separate, transformed copy can reuse them.
    return {'signature': session.geometry_signature(mesh, Matrix.Identity(4)),
            'vertex_count': len(mesh.vertices),
            'triangle_count': len(mesh.loop_triangles),
            'uv_map': mesh.uv_layers.active.name if mesh.uv_layers.active else ''}


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    return value


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'{label} must contain {size} values.')
    return [_number(item, label) for item in value]


def _validate_guides(records, mesh):
    if not isinstance(records, list):
        raise ValueError('The guide file must contain a guides list.')
    mesh.calc_loop_triangles()
    triangles = {tuple(triangle.vertices) for triangle in mesh.loop_triangles}
    fields = {'vertices', 'barycentric', 'tip_vertices', 'tip_barycentric',
              'has_tip', 'direction', 'strength', 'mirror_pair'}
    result = []
    for index, record in enumerate(records, 1):
        label = f'Guide {index}'
        if not isinstance(record, dict) or not fields.issubset(record):
            raise ValueError(f'{label} is missing required guide properties.')
        if type(record['has_tip']) is not bool:
            raise ValueError(f'{label} has_tip must be true or false.')
        if not isinstance(record['mirror_pair'], str):
            raise ValueError(f'{label} mirror_pair must be a string.')
        clean = {'has_tip': record['has_tip'], 'mirror_pair': record['mirror_pair']}
        for prefix in ('', 'tip_'):
            ids = _vector(record[prefix + 'vertices'], 3, f'{label} {prefix}vertices')
            if any(type(v) is not int or not 0 <= v < len(mesh.vertices) for v in ids):
                raise ValueError(f'{label} contains an invalid vertex index.')
            weights = _vector(record[prefix + 'barycentric'], 3, f'{label} {prefix}barycentric')
            if any(w < 0 or w > 1 for w in weights):
                raise ValueError(f'{label} anchor weights must lie between 0 and 1.')
            if not prefix or record['has_tip']:
                if tuple(ids) not in triangles:
                    raise ValueError(f'{label} anchor does not match a mesh triangle.')
                if abs(sum(weights) - 1.0) > 1e-5:
                    raise ValueError(f'{label} anchor weights must sum to 1.')
            clean[prefix + 'vertices'] = ids
            clean[prefix + 'barycentric'] = weights
        direction = _vector(record['direction'], 2, f'{label} direction')
        if abs(math.hypot(*direction) - 1.0) > 1e-5:
            raise ValueError(f'{label} direction must have unit length.')
        strength = _number(record['strength'], f'{label} strength')
        if not 0 <= strength <= 1:
            raise ValueError(f'{label} strength must lie between 0 and 1.')
        clean.update(direction=direction, strength=strength)
        result.append(clean)
    pairs = Counter(record['mirror_pair'] for record in result if record['mirror_pair'])
    if any(count != 2 for count in pairs.values()):
        raise ValueError('Each mirror pair must link exactly two guides.')
    return result


def _validate_idle(obj):
    if session.ACTIVE is not None:
        raise ValueError('Exit Flow Paint Mode before importing or exporting guides.')
    if obj is None or obj.type != 'MESH' or obj.mode != 'OBJECT':
        raise ValueError('Select one mesh in Object Mode.')
    if not obj.data.uv_layers.active:
        raise ValueError('The mesh needs an active UV map.')


def _records(mesh):
    return [session.PaintSession._record_data(stroke) for stroke in mesh.warpflow.strokes]


def _replace_records(mesh, records):
    mesh.warpflow.strokes.clear()
    for record in records:
        stroke = mesh.warpflow.strokes.add()
        for name, value in record.items():
            setattr(stroke, name, value)


def export_guides(obj, filepath):
    """Save every completed guide in order; replace a file only after writing."""
    _validate_idle(obj)
    mesh = obj.data
    if mesh.warpflow.strokes and mesh.warpflow.geometry_signature != session.geometry_signature(mesh, obj.matrix_world):
        raise ValueError('Geometry, transform or UVs changed since painting. Restore them before exporting guides.')
    payload = {'format': FORMAT, 'version': VERSION, 'mesh': _mesh_info(mesh),
               'guides': _validate_guides(_records(mesh), mesh)}
    if not filepath:
        raise ValueError('Choose a guide file path.')
    output = os.path.abspath(bpy.path.abspath(bpy.path.ensure_ext(os.fspath(filepath), '.json')))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         prefix='.warpflow-guides-', suffix='.json',
                                         dir=os.path.dirname(output), delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary_path, output)
        temporary_path = None
        return output
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def import_guides(context, filepath, mode='APPEND', progress=None):
    """Validate, import, and replay guides as one undoable operator transaction.

    Only guide metadata is exchanged. Replay uses the destination scene's
    sharpness and distance settings, just like entering Flow Paint Mode.
    """
    obj = context.object
    _validate_idle(obj)
    session.validate_object(obj)
    if mode not in {'APPEND', 'REPLACE'}:
        raise ValueError('Choose Append or Replace for guide import.')
    if not filepath:
        raise ValueError('Choose a guide file to import.')
    try:
        with open(bpy.path.abspath(os.fspath(filepath)), encoding='utf-8-sig') as stream:
            payload = json.load(stream)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ValueError('The file is not valid Warpflow guide JSON.') from exc
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError('This is not a Warpflow guide file.')
    if type(payload.get('version')) is not int or payload['version'] != VERSION:
        raise ValueError('Unsupported Warpflow guide file version.')
    mesh = obj.data
    expected = _mesh_info(mesh)
    source = payload.get('mesh')
    if not isinstance(source, dict) or any(source.get(key) != expected[key]
                                          for key in ('signature', 'vertex_count', 'triangle_count')):
        raise ValueError('Guides require matching mesh geometry, vertex order, topology and active UVs.')
    imported = _validate_guides(payload.get('guides'), mesh)
    signature = session.geometry_signature(mesh, obj.matrix_world)
    previous_signature = mesh.warpflow.geometry_signature
    previous_records = _records(mesh)
    if mode == 'APPEND' and previous_records:
        if previous_signature != signature:
            raise ValueError('Existing guides are stale. Use Replace to import a new guide history.')
        _validate_guides(previous_records, mesh)
    # A file can be appended repeatedly without linking unrelated pair copies.
    pair_ids = {record['mirror_pair']: uuid.uuid4().hex for record in imported if record['mirror_pair']}
    for record in imported:
        if record['mirror_pair']:
            record['mirror_pair'] = pair_ids[record['mirror_pair']]
    combined = previous_records + imported if mode == 'APPEND' else imported
    attr = mesh.attributes.get(session.ATTRIBUTE)
    if attr is not None and (attr.domain != 'POINT' or attr.data_type != 'FLOAT_COLOR'):
        raise ValueError('Existing Warpflow attribute must be Point / Float Color; rename it before importing.')
    previous_colors = None
    if attr is not None:
        previous_colors = np.empty(len(attr.data) * 4, dtype=np.float32)
        attr.data.foreach_get('color', previous_colors)
    active_index = mesh.color_attributes.active_color_index
    render_index = mesh.color_attributes.render_color_index
    prepared = None
    try:
        _replace_records(mesh, combined)
        mesh.warpflow.geometry_signature = signature
        prepared = session.PaintSession(context, progress=progress)
    except Exception:
        _replace_records(mesh, previous_records)
        mesh.warpflow.geometry_signature = previous_signature
        current = mesh.color_attributes.get(session.ATTRIBUTE)
        if previous_colors is None:
            if current is not None:
                mesh.color_attributes.remove(current)
        elif current is not None:
            current.data.foreach_set('color', previous_colors)
        if active_index >= 0:
            mesh.color_attributes.active_color_index = active_index
        if render_index >= 0:
            mesh.color_attributes.render_color_index = render_index
        mesh.update()
        raise
    finally:
        if prepared is not None:
            prepared.close()
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return len(imported)
