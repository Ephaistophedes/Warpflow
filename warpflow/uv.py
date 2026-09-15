"""UV connectivity and local tangent frames, computed once per paint session.

A curved UV island cannot have one constant 3D tangent frame. We identify its
face connectivity once, then build triangle frames and area-weighted vertex
frames *within* each island. POINT colors have one value at a shared vertex:
the dominant-area island owns that vertex's display frame. Distinct UV seam
corners cannot store independent vectors without splitting the mesh vertices.
No deprecated MeshUVLoopLayer.data or Mesh.vertex_colors API is used.
"""

from dataclasses import dataclass

import numpy as np


def _read(collection, field, width, dtype=np.float64):
    result = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(field, result)
    return result.reshape((-1, width))


def _unit(vectors):
    length = np.linalg.norm(vectors, axis=1)
    return vectors / np.maximum(length[:, None], 1e-30)


@dataclass
class UVData:
    vertex_tangents: np.ndarray
    vertex_bitangents: np.ndarray
    triangle_tangents: np.ndarray
    triangle_bitangents: np.ndarray
    triangle_islands: np.ndarray
    vertex_islands: np.ndarray
    island_count: int
    warnings: list
    uv_layer_name: str
    loop_uvs: np.ndarray

    def direction_to_uv(self, triangle_index, local_direction):
        """Project an object-local direction into its hit triangle's UV frame.

        Return a normalized 2-vector, or None for a zero/normal-only drag.
        Mirrored islands retain the handedness of their actual UV coordinates.
        """
        direction = np.asarray(local_direction, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            return None
        uv = np.array((np.dot(direction, self.triangle_tangents[triangle_index]),
                       np.dot(direction, self.triangle_bitangents[triangle_index])))
        length = float(np.linalg.norm(uv))
        if length <= max(1e-12, float(np.linalg.norm(direction)) * 1e-10):
            return None
        return uv / length


def _polygon_islands(mesh, loop_uvs, tolerance=1e-6):
    """Union faces only when a topological edge has matching endpoint UVs.

    An artist's seam flag is not sufficient: marking a seam does not split the
    UV map until it is unwrapped. Conversely, split UVs count without a flag.
    Non-manifold edges compare every incident UV edge against representatives;
    ordinary manifold meshes take linear time in the number of face corners.
    """
    count = len(mesh.polygons)
    parent = np.arange(count, dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first, second):
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    loop_vertices = _read(mesh.loops, "vertex_index", 1, np.int32).ravel()
    loop_edges = _read(mesh.loops, "edge_index", 1, np.int32).ravel()
    edges = {}
    seam_vertices = set()
    for face in mesh.polygons:
        start, total = face.loop_start, face.loop_total
        for corner in range(start, start + total):
            following = start + (corner - start + 1) % total
            first, second = loop_uvs[corner], loop_uvs[following]
            if loop_vertices[corner] > loop_vertices[following]:
                first, second = second, first
            uv_edge = np.concatenate((first, second))
            representatives = edges.setdefault(int(loop_edges[corner]), [])
            for other_face, other_uv in representatives:
                if np.max(np.abs(uv_edge - other_uv)) <= tolerance:
                    union(face.index, other_face)
                    break
            else:
                if representatives:
                    seam_vertices.update((int(loop_vertices[corner]), int(loop_vertices[following])))
                representatives.append((face.index, uv_edge))
    roots = np.fromiter((find(i) for i in range(count)), dtype=np.int32, count=count)
    _, islands = np.unique(roots, return_inverse=True)
    return islands.astype(np.int32), seam_vertices


def build_uv_data(mesh):
    """Validate an unwrap and cache UV islands and local tangent bases.

    UVs outside the first tile are usable for painting; export separately
    rejects those because v1 produces one PNG. Degenerate UV triangles are
    rejected here: inventing their tangent would silently encode wrong flow.
    """
    layer = mesh.uv_layers.active
    if layer is None:
        raise ValueError("Warpflow needs an active UV map. UV unwrap the mesh first.")
    if not len(mesh.vertices) or not len(mesh.polygons):
        raise ValueError("Warpflow needs a mesh with faces, not only points or edges.")
    mesh.calc_loop_triangles()
    positions = _read(mesh.vertices, "co", 3)
    uv = _read(layer.uv, "vector", 2)
    if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(positions)):
        raise ValueError("Mesh coordinates or UVs contain non-finite values. Repair the mesh and unwrap.")
    triangles = _read(mesh.loop_triangles, "vertices", 3, np.int32)
    corners = _read(mesh.loop_triangles, "loops", 3, np.int32)
    face_ids = _read(mesh.loop_triangles, "polygon_index", 1, np.int32).ravel()
    points = positions[triangles]
    texcoords = uv[corners]
    edge1, edge2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    duv1, duv2 = texcoords[:, 1] - texcoords[:, 0], texcoords[:, 2] - texcoords[:, 0]
    determinant = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    uv_scale = np.linalg.norm(duv1, axis=1) * np.linalg.norm(duv2, axis=1)
    bad = np.abs(determinant) <= np.maximum(uv_scale * 1e-12, 1e-24)
    if np.any(bad):
        face = int(face_ids[np.flatnonzero(bad)[0]])
        raise ValueError(f"Degenerate UV triangle on face {face + 1}. Unwrap collapsed UV faces before painting or export.")
    normals_raw = np.cross(edge1, edge2)
    double_area = np.linalg.norm(normals_raw, axis=1)
    geometric_scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    if np.any(double_area <= np.maximum(geometric_scale * 1e-12, 1e-30)):
        raise ValueError("The mesh contains zero-area triangles. Merge or remove degenerate faces first.")
    normals = normals_raw / double_area[:, None]
    tangent_raw = (edge1 * duv2[:, 1, None] - edge2 * duv1[:, 1, None]) / determinant[:, None]
    tangent = _unit(tangent_raw)
    bitangent = np.cross(normals, tangent) * np.sign(determinant)[:, None]
    face_islands, seam_vertices = _polygon_islands(mesh, uv)
    tri_islands = face_islands[face_ids]
    island_count = int(face_islands.max()) + 1

    # Aggregate by (vertex, island), never averaging unrelated UV seam frames.
    flat_vertices = triangles.ravel()
    pair_keys = flat_vertices.astype(np.int64) * island_count + np.repeat(tri_islands, 3)
    unique_pairs, inverse = np.unique(pair_keys, return_inverse=True)
    pair_vertices = unique_pairs // island_count
    pair_islands = unique_pairs % island_count
    pair_area = np.bincount(inverse, weights=np.repeat(double_area, 3))
    order = np.lexsort((pair_islands, -pair_area, pair_vertices))
    sorted_vertices = pair_vertices[order]
    first = np.r_[True, sorted_vertices[1:] != sorted_vertices[:-1]]
    chosen = order[first]
    owned_vertices = pair_vertices[chosen]
    group_t = np.zeros((len(unique_pairs), 3))
    group_b = np.zeros_like(group_t)
    group_n = np.zeros_like(group_t)
    weight = double_area[:, None]
    np.add.at(group_t, inverse, np.repeat(tangent * weight, 3, axis=0))
    np.add.at(group_b, inverse, np.repeat(bitangent * weight, 3, axis=0))
    np.add.at(group_n, inverse, np.repeat(normals * weight, 3, axis=0))

    # A highly folded neighborhood can cancel its average. The largest incident
    # triangle supplies a deterministic, valid fallback instead of a zero frame.
    triangle_repeat = np.repeat(np.arange(len(triangles)), 3)
    fallback_order = np.lexsort((triangle_repeat, -np.repeat(double_area, 3), inverse))
    fallback_inverse = inverse[fallback_order]
    group_first = np.r_[True, fallback_inverse[1:] != fallback_inverse[:-1]]
    fallback_tri = np.empty(len(unique_pairs), dtype=np.int32)
    fallback_tri[fallback_inverse[group_first]] = triangle_repeat[fallback_order[group_first]]
    chosen_tri = fallback_tri[chosen]
    n = group_n[chosen]
    bad_n = np.linalg.norm(n, axis=1) <= pair_area[chosen] * 1e-8
    n[bad_n] = normals[chosen_tri[bad_n]]
    n = _unit(n)
    t = group_t[chosen]
    t -= n * np.sum(t * n, axis=1)[:, None]
    bad_t = np.linalg.norm(t, axis=1) <= pair_area[chosen] * 1e-8
    if np.any(bad_t):
        n[bad_t] = normals[chosen_tri[bad_t]]
        t[bad_t] = tangent[chosen_tri[bad_t]]
    t = _unit(t)
    handedness = np.sign(np.sum(np.cross(n, t) * group_b[chosen], axis=1))
    handedness[handedness == 0] = np.sign(determinant[chosen_tri[handedness == 0]])
    b = np.cross(n, t) * handedness[:, None]
    vertex_t = np.tile((1.0, 0.0, 0.0), (len(positions), 1))
    vertex_b = np.tile((0.0, 1.0, 0.0), (len(positions), 1))
    vertex_islands = np.full(len(positions), -1, dtype=np.int32)
    vertex_t[owned_vertices] = t
    vertex_b[owned_vertices] = b
    vertex_islands[owned_vertices] = pair_islands[chosen]
    warning_list = []
    # A cut cylinder still forms one UV island, but its seam vertices have two
    # UV corners. Include those as well as vertices shared by distinct islands.
    seam_vertices.update(np.flatnonzero(np.bincount(pair_vertices, minlength=len(positions)) > 1).tolist())
    seam_count = len(seam_vertices)
    if seam_count:
        warning_list.append(f"{seam_count} vertices have split UV seam corners. POINT flow uses the dominant-area island frame; split mesh vertices at seams for independent corner directions.")
    loose_count = int(np.count_nonzero(vertex_islands < 0))
    if loose_count:
        warning_list.append(f"{loose_count} loose vertices have no UV surface; their arrows use the local XY frame and they do not bake.")
    return UVData(vertex_t, vertex_b, tangent, bitangent, tri_islands,
                  vertex_islands, island_count, warning_list, layer.name, uv)
