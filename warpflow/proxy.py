"""Topology-checked, component-wise collapsed meshes for live flow previews.

Blender's native Decimate COLLAPSE (quadric-error edge collapse) runs once during
session setup. Barycentric BVH transfer is also precomputed once; every drag
preview thereafter costs only three indexed reads per full-resolution vertex.
Neither decimation nor transfer ever combines disconnected surface components.
We reject a decimation if its Euler characteristic, boundary count, manifoldness,
or transfer continuity changes. Such a component stays at full resolution. This
conservative fallback matters for holes, thin folded sheets, and nonmanifolds.

The preview is approximate. The original mesh solver is always used on commit;
no decimated result becomes the persisted color attribute at mouse release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .interpolation import normalize_vectors


def _edges(triangles):
    if len(triangles) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int32)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def _topology_signature(n_vertices, triangles):
    """Return a manifold surface fingerprint; None means unsafe to simplify."""
    triangles = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    if not len(triangles) or len(np.unique(triangles)) != n_vertices:
        return None  # Preserve loose geometry verbatim.
    edges, counts = _edges(triangles)
    if np.any(counts > 2) or np.any(triangles[:, 0] == triangles[:, 1]) or np.any(triangles[:, 1] == triangles[:, 2]) or np.any(triangles[:, 2] == triangles[:, 0]):
        return None
    parent = np.arange(n_vertices, dtype=np.int32)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[root(int(a))] = root(int(b))
    connected_count = len({root(i) for i in range(n_vertices)})
    boundary_edges = edges[counts == 1]
    if len(boundary_edges):
        boundary_degree = np.bincount(boundary_edges.ravel(), minlength=n_vertices)
        if np.any((boundary_degree != 0) & (boundary_degree != 2)):
            return None  # Pinched boundary / nonmanifold vertex.
        parent = np.arange(n_vertices, dtype=np.int32)
        for a, b in boundary_edges:
            parent[root(int(a))] = root(int(b))
        boundary_count = len({root(int(i)) for i in np.unique(boundary_edges)})
    else:
        boundary_count = 0
    return (n_vertices - len(edges) + len(triangles), boundary_count, connected_count)


def _barycentric(point, triangle):
    """Clamped barycentric coordinates at a nearest point on a triangle."""
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, np.asarray(point) - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1.0e-30:
        result = np.zeros(3)
        result[int(np.argmin(np.linalg.norm(triangle - point, axis=1)))] = 1.0
        return result
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    result = np.clip([1.0 - v - w, v, w], 0.0, 1.0)
    return result / max(float(np.sum(result)), 1.0e-30)


def _nearest_on_triangles(point, triangles):
    """Vectorized closest points for a small topologically restricted candidate set."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    length_squared = np.sum(normal * normal, axis=1)
    height = np.divide(np.sum((point - a) * normal, axis=1), length_squared,
                       out=np.zeros(len(triangles)), where=length_squared > 1.0e-30)
    projected = point - height[:, None] * normal
    ap = projected - a
    d00, d01, d11 = np.sum(ab * ab, axis=1), np.sum(ab * ac, axis=1), np.sum(ac * ac, axis=1)
    d20, d21 = np.sum(ap * ab, axis=1), np.sum(ap * ac, axis=1)
    denominator = d00 * d11 - d01 * d01
    v = np.divide(d11 * d20 - d01 * d21, denominator, out=np.full(len(triangles), -1.0), where=denominator > 1.0e-30)
    w = np.divide(d00 * d21 - d01 * d20, denominator, out=np.full(len(triangles), -1.0), where=denominator > 1.0e-30)
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    starts = triangles
    segments = triangles[:, [1, 2, 0]] - starts
    denominator = np.sum(segments * segments, axis=2)
    fractions = np.divide(np.sum((point - starts) * segments, axis=2), denominator,
                          out=np.zeros_like(denominator), where=denominator > 1.0e-30)
    edge_points = starts + np.clip(fractions, 0, 1)[:, :, None] * segments
    edge_index = np.argmin(np.sum((edge_points - point) ** 2, axis=2), axis=1)
    closest = edge_points[np.arange(len(triangles)), edge_index]
    closest[inside] = projected[inside]
    best = int(np.argmin(np.sum((closest - point) ** 2, axis=1)))
    return closest[best], best


def _decimate_component(vertices, triangles, ratio):
    """Evaluate an isolated temporary datablock; user objects are never changed."""
    import bpy

    mesh = bpy.data.meshes.new(".WarpflowPreviewSource")
    obj = None
    evaluated = None
    try:
        mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
        mesh.update()
        obj = bpy.data.objects.new(".WarpflowPreviewSource", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.hide_render = True
        modifier = obj.modifiers.new("Warpflow preview collapse", 'DECIMATE')
        modifier.decimate_type = 'COLLAPSE'
        modifier.ratio = float(ratio)
        modifier.use_collapse_triangulate = True
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        result = evaluated.to_mesh()
        result.calc_loop_triangles()
        positions = np.empty(len(result.vertices) * 3, dtype=np.float64)
        result.vertices.foreach_get("co", positions)
        faces = np.empty(len(result.loop_triangles) * 3, dtype=np.int32)
        result.loop_triangles.foreach_get("vertices", faces)
        return positions.reshape((-1, 3)), faces.reshape((-1, 3))
    finally:
        if evaluated is not None:
            evaluated.to_mesh_clear()
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _tree(vertices, triangles):
    from mathutils.bvhtree import BVHTree

    return BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)


def _transfer(vertices, proxy_vertices, proxy_triangles, tree):
    indices = np.empty((len(vertices), 3), dtype=np.int32)
    weights = np.empty((len(vertices), 3), dtype=np.float64)
    faces = np.empty(len(vertices), dtype=np.int32)
    for index, position in enumerate(vertices):
        nearest, _normal, face, _distance = tree.find_nearest(position)
        if face is None:
            raise ValueError("Preview surface has no nearest triangle")
        faces[index] = face
        indices[index] = proxy_triangles[face]
        weights[index] = _barycentric(nearest, proxy_vertices[proxy_triangles[face]])
    return indices, weights, faces


def _orientation_preserved(full_vertices, full_triangles, proxy_vertices, proxy_triangles, face_map):
    """Reject nearest-point mappings to the back of a nearby folded sheet."""
    full_points = full_vertices[full_triangles]
    face_normals = np.cross(full_points[:, 1] - full_points[:, 0], full_points[:, 2] - full_points[:, 0])
    vertex_normals = np.zeros_like(full_vertices)
    for corner in range(3):
        np.add.at(vertex_normals, full_triangles[:, corner], face_normals)
    proxy_points = proxy_vertices[proxy_triangles[face_map]]
    mapped_normals = np.cross(proxy_points[:, 1] - proxy_points[:, 0], proxy_points[:, 2] - proxy_points[:, 0])
    denominator = np.linalg.norm(vertex_normals, axis=1) * np.linalg.norm(mapped_normals, axis=1)
    cosine = np.divide(np.sum(vertex_normals * mapped_normals, axis=1), denominator,
                       out=np.ones(len(full_vertices)), where=denominator > 1.0e-30)
    # Slightly opposing corner averages can result from sharp edges. A strong
    # reversal is evidence of transfer to the opposite side and cannot be used.
    return not np.any(cosine < -0.05)


def _continuous_transfer(full_triangles, proxy_triangles, face_map,
                         full_vertices=None, proxy_vertices=None, tree=None,
                         transfer_points=None):
    """Adjacent original vertices must map to nearby proxy topology.

    A Euclidean nearest query alone can jump between close folds. Require the
    mapped triangles at both ends of every original edge to share a proxy vertex
    or have a connected route through a small neighborhood of that original
    edge. The latter accommodates skinny collapsed triangles on planar meshes.
    Nearby folds without a nearby surface route fail this check; the component
    then retains its original geometry.
    """
    edges, _counts = _edges(full_triangles)
    exceptional = []
    # Chunking avoids a large E x 3 x 3 boolean allocation on dense input meshes.
    for start in range(0, len(edges), 100000):
        chunk = edges[start:start + 100000]
        a = proxy_triangles[face_map[chunk[:, 0]]]
        b = proxy_triangles[face_map[chunk[:, 1]]]
        shared = np.any(a[:, :, None] == b[:, None, :], axis=(1, 2))
        exceptional.extend(chunk[~shared])
    if not exceptional:
        return True
    if full_vertices is None or proxy_vertices is None or tree is None:
        return False
    neighbors = [[] for _ in proxy_triangles]
    edge_faces = {}
    for face, triangle in enumerate(proxy_triangles):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            if edge in edge_faces:
                other = edge_faces[edge]
                neighbors[face].append((other, *edge))
                neighbors[other].append((face, *edge))
            else:
                edge_faces[edge] = face
    scale = max(float(np.linalg.norm(np.ptp(full_vertices, axis=0))), 1.0e-10)
    tolerance = scale * 1.0e-6
    for u, v in exceptional:
        start_face, end_face = int(face_map[u]), int(face_map[v])
        midpoint = (full_vertices[u] + full_vertices[v]) * 0.5
        radius = float(np.linalg.norm(full_vertices[u] - full_vertices[v])) * 1.1 + tolerance
        if transfer_points is not None:
            radius += max(float(np.linalg.norm(transfer_points[u] - full_vertices[u])),
                          float(np.linalg.norm(transfer_points[v] - full_vertices[v])))
        active = {item[2] for item in tree.find_nearest_range(midpoint, radius)}
        if start_face not in active or end_face not in active:
            return False
        visited, pending = {start_face}, [start_face]
        found = False
        while pending and not found:
            current = pending.pop()
            for other, a, b in neighbors[current]:
                if other in visited or other not in active:
                    continue
                segment = proxy_vertices[b] - proxy_vertices[a]
                fraction = np.clip(np.dot(midpoint - proxy_vertices[a], segment)
                                   / max(float(np.dot(segment, segment)), 1.0e-30), 0.0, 1.0)
                crossing = proxy_vertices[a] + fraction * segment
                if np.linalg.norm(crossing - midpoint) > radius:
                    continue  # Shared edge lies outside the local neighborhood.
                if other == end_face:
                    found = True
                    break
                visited.add(other)
                pending.append(other)
        if not found:
            return False
    return True


@dataclass
class ProxyData:
    vertices: np.ndarray
    triangles: np.ndarray
    components: np.ndarray
    full_vertices: np.ndarray
    full_components: np.ndarray
    transfer_indices: np.ndarray
    transfer_weights: np.ndarray
    direct_vertices: np.ndarray
    component_maps: dict = field(default_factory=dict, repr=False)
    stats: dict = field(default_factory=dict)

    @property
    def is_proxy(self):
        return len(self.vertices) < len(self.full_vertices)

    @property
    def reduced(self):
        return self.is_proxy

    def map_source(self, full_indices, barycentric_weights=None):
        """Map an original single-point barycentric source to its own component."""
        indices = np.asarray(full_indices, dtype=np.int32).reshape(-1)
        if barycentric_weights is None:
            weights = np.full(len(indices), 1.0 / len(indices))
        else:
            weights = np.asarray(barycentric_weights, dtype=np.float64).reshape(-1)
        if len(indices) != len(weights) or not len(indices):
            raise ValueError("Source vertices and weights must be nonempty and match")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
            raise ValueError("Source barycentric weights must be finite and nonnegative")
        weights = weights / np.sum(weights)
        labels = self.full_components[indices]
        if np.any(labels != labels[0]):
            raise ValueError("A stroke source cannot bridge disconnected components")
        mapped = self.direct_vertices[indices]
        if np.all(mapped >= 0):
            return mapped, weights
        label = int(labels[0])
        tree, triangles, vertex_indices, vertex_faces = self.component_maps[label]
        point = np.sum(self.full_vertices[indices] * weights[:, None], axis=0)
        if tree is None:
            nearest = vertex_indices[np.argmin(np.linalg.norm(self.vertices[vertex_indices] - point, axis=1))]
            return np.asarray([nearest], dtype=np.int32), np.asarray([1.0])
        # The original triangle corners already have validated surface transfers.
        # Restrict its source query to their neighboring proxy faces to avoid a
        # fresh unconstrained nearest query jumping to a nearby folded sheet.
        neighborhood = np.unique(self.transfer_indices[indices])
        candidates = sorted({face for vertex in neighborhood for face in vertex_faces[int(vertex)]})
        if not candidates:
            raise ValueError("Could not map source to preview surface")
        candidate_triangles = triangles[candidates]
        nearest, candidate_index = _nearest_on_triangles(point, self.vertices[candidate_triangles])
        mapped = candidate_triangles[candidate_index]
        return mapped, _barycentric(nearest, self.vertices[mapped])

    def upsample(self, proxy_field):
        """O(V) cached barycentric interpolation back to original vertices."""
        values = np.asarray(proxy_field, dtype=np.float64)
        if values.shape != (len(self.vertices), 2):
            raise ValueError("Preview field must have one 2D vector per proxy vertex")
        contributions = values[self.transfer_indices]
        result = np.sum(contributions * self.transfer_weights[:, :, None], axis=1)
        strongest = np.argmax(self.transfer_weights, axis=1)
        fallback = contributions[np.arange(len(result)), strongest]
        return normalize_vectors(result, fallback)


def build_proxy(obj, full_vertices, full_triangles, full_components, target_vertices=5000, progress=None):
    """Build a collapsed preview and cached transfer once, with safe fallbacks.

    ``obj`` identifies the session object for callers; geometry arguments are the
    authoritative solve-space positions (normally object-local coordinates).
    A returned unreduced proxy is valid and signals that topology preservation
    required full-resolution preview. No meshes or objects are left in the file.
    """
    started = time.perf_counter()
    vertices = np.asarray(full_vertices, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(full_triangles, dtype=np.int32).reshape((-1, 3))
    components = np.asarray(full_components, dtype=np.int32)
    if components.shape != (len(vertices),):
        raise ValueError("Component labels must have one entry per vertex")
    ratio = min(1.0, max(4, int(target_vertices)) / max(1, len(vertices)))
    output_vertices, output_triangles, output_components = [], [], []
    transfer_indices = np.empty((len(vertices), 3), dtype=np.int32)
    transfer_weights = np.zeros((len(vertices), 3), dtype=np.float64)
    direct_vertices = np.full(len(vertices), -1, dtype=np.int32)
    maps, fallbacks = {}, []
    offset = 0
    labels = np.unique(components)
    full_to_local = np.empty(len(vertices), dtype=np.int32)
    for component_number, label in enumerate(labels):
        if progress is not None:
            progress(component_number / max(1, len(labels)))
        full_indices = np.flatnonzero(components == label)
        local_vertices = vertices[full_indices]
        full_to_local[full_indices] = np.arange(len(full_indices))
        face_mask = components[triangles[:, 0]] == label if len(triangles) else np.zeros(0, dtype=bool)
        local_triangles = full_to_local[triangles[face_mask]]
        reduced_vertices, reduced_triangles = local_vertices, local_triangles
        transfer = None
        component_tree = None
        if ratio < 0.95 and len(local_vertices) > 16:
            signature = _topology_signature(len(local_vertices), local_triangles)
            if signature is None:
                fallbacks.append((int(label), "nonmanifold or loose geometry"))
            else:
                try:
                    candidate_vertices, candidate_triangles = _decimate_component(local_vertices, local_triangles, ratio)
                    if _topology_signature(len(candidate_vertices), candidate_triangles) != signature:
                        raise ValueError("collapse changed connected topology or a hole boundary")
                    if len(candidate_vertices) < len(local_vertices):
                        candidate_tree = _tree(candidate_vertices, candidate_triangles)
                        candidate_transfer = _transfer(local_vertices, candidate_vertices, candidate_triangles, candidate_tree)
                        if not _orientation_preserved(local_vertices, local_triangles, candidate_vertices,
                                                       candidate_triangles, candidate_transfer[2]):
                            raise ValueError("nearest-surface transfer reversed surface orientation")
                        transfer_points = np.sum(candidate_vertices[candidate_transfer[0]]
                                                 * candidate_transfer[1][:, :, None], axis=1)
                        if not _continuous_transfer(local_triangles, candidate_triangles, candidate_transfer[2],
                                                    local_vertices, candidate_vertices, candidate_tree, transfer_points):
                            raise ValueError("nearest-surface transfer jumped across a fold")
                        reduced_vertices, reduced_triangles = candidate_vertices, candidate_triangles
                        component_tree = candidate_tree
                        transfer = candidate_transfer
                except Exception as exc:
                    fallbacks.append((int(label), str(exc)))
        count = len(reduced_vertices)
        if transfer is None:
            mapped = np.arange(offset, offset + count, dtype=np.int32)
            transfer_indices[full_indices] = mapped[:, None]
            transfer_weights[full_indices, 0] = 1.0
            direct_vertices[full_indices] = mapped
        else:
            transfer_indices[full_indices] = transfer[0] + offset
            transfer_weights[full_indices] = transfer[1]
        if component_tree is None and len(reduced_triangles):
            component_tree = _tree(reduced_vertices, reduced_triangles)
        global_triangles = reduced_triangles + offset
        vertex_faces = {index: [] for index in range(offset, offset + count)}
        if transfer is not None:
            for face, triangle in enumerate(global_triangles):
                for vertex in triangle:
                    vertex_faces[int(vertex)].append(face)
        maps[int(label)] = (component_tree, global_triangles, np.arange(offset, offset + count, dtype=np.int32), vertex_faces)
        output_vertices.append(reduced_vertices)
        output_triangles.append(global_triangles)
        output_components.append(np.full(count, label, dtype=np.int32))
        offset += count
    if progress is not None:
        progress(1.0)
    return ProxyData(
        vertices=np.concatenate(output_vertices) if output_vertices else np.empty((0, 3)),
        triangles=np.concatenate(output_triangles) if output_triangles else np.empty((0, 3), dtype=np.int32),
        components=np.concatenate(output_components) if output_components else np.empty(0, dtype=np.int32),
        full_vertices=vertices, full_components=components,
        transfer_indices=transfer_indices, transfer_weights=transfer_weights,
        direct_vertices=direct_vertices, component_maps=maps,
        stats={"full_vertices": len(vertices), "proxy_vertices": offset,
               "setup_seconds": time.perf_counter() - started, "fallbacks": fallbacks},
    )
