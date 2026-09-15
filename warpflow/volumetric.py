"""Volumetric, interior-constrained distances for closed triangle meshes.

The surface solver works on the mesh's two-dimensional manifold.  This module
instead voxelizes its enclosed volume and solves shortest paths through the
occupied cells.  It uses a 26-neighbour, Euclidean-weighted grid graph: a
simple, robust discretization of a volumetric distance field.  Resolution is
deliberately exposed to artists because a finer grid follows narrow passages
more closely at the cost of setup and source-query time.

This module needs Blender's BVH for efficient scanline voxelization, so it is
kept separate from :mod:`geodesic`, whose numerical surface solver can run in
ordinary Python for its unit tests.
"""

from __future__ import annotations

import heapq
import itertools
import time

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _surface_edges(triangles):
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _closed_manifold(triangles):
    if not len(triangles):
        return False
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


class VolumetricSolver:
    """Reusable 3D distance field for a closed, manifold triangle mesh.

    ``distances(indices, weights)`` has the same public contract as the
    surface solver.  Distances are finite only on mesh vertices whose mapped
    interior cells share the source's volumetric component.
    """

    def __init__(self, vertices, triangles, resolution=48, progress=None):
        started = time.perf_counter()
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or not np.all(np.isfinite(self.vertices))):
            raise ValueError('vertices must be a finite array with shape (N, 3)')
        if (self.triangles.ndim != 2 or self.triangles.shape[1] != 3
                or np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices))):
            raise ValueError('triangles must contain valid vertex triples')
        if not _closed_manifold(self.triangles):
            raise ValueError('Volumetric distance requires a closed, manifold mesh.')
        self.vertex_count = len(self.vertices)
        self.resolution = int(resolution)
        if not 24 <= self.resolution <= 120:
            raise ValueError('Volumetric resolution must be between 24 and 120.')
        self._progress = progress or (lambda value: None)
        self.stats = {'vertex_count': self.vertex_count, 'triangle_count': len(self.triangles),
                      'factorization_count': 0, 'solve_count': 0, 'grid_solve_count': 0}
        self._progress(0.04)
        self._setup_grid()
        self._progress(0.94)
        self.backend = ('Volumetric voxel distance field '
                        f'({self.grid_shape[0]}×{self.grid_shape[1]}×{self.grid_shape[2]}, 26-neighbor)')
        self.stats['backend'] = self.backend
        self.stats['setup_seconds'] = time.perf_counter() - started
        self._progress(1.0)
        self._progress = None

    def _setup_grid(self):
        bounds_min = self.vertices.min(axis=0)
        bounds_max = self.vertices.max(axis=0)
        extent = bounds_max - bounds_min
        longest = float(np.max(extent))
        if longest <= 1e-12:
            raise ValueError('Volumetric distance requires a mesh with nonzero size.')
        # The selectable resolution is the number of cells over the longest
        # object dimension.  One-cell padding makes ray starts unambiguous.
        self.cell_size = longest / self.resolution
        counts = np.maximum(np.ceil(extent / self.cell_size).astype(np.int32) + 2, 4)
        self.grid_shape = tuple(int(v) for v in counts)
        self.grid_origin = bounds_min - self.cell_size
        total = int(np.prod(counts))
        max_cells = 1_800_000
        if total > max_cells:
            raise ValueError('Volumetric grid is too large; lower Volumetric Resolution.')
        self._bvh = BVHTree.FromPolygons(self.vertices.tolist(), self.triangles.tolist(),
                                         all_triangles=True, epsilon=self.cell_size * 1e-6)
        inside = np.zeros(self.grid_shape, dtype=bool)
        direction = Vector((1.0, 0.000173, 0.000317)).normalized()
        ray_start_x = float(self.grid_origin[0] - self.cell_size)
        ray_length = float(extent[0] + 4.0 * self.cell_size)
        epsilon = self.cell_size * 1e-5
        nx, ny, nz = self.grid_shape
        for iy in range(ny):
            y = self.grid_origin[1] + (iy + 0.5) * self.cell_size
            for iz in range(nz):
                z = self.grid_origin[2] + (iz + 0.5) * self.cell_size
                origin = Vector((ray_start_x, float(y), float(z)))
                hits = []
                travelled = 0.0
                while travelled < ray_length:
                    location, _normal, _index, distance = self._bvh.ray_cast(origin, direction,
                                                                               ray_length - travelled)
                    if location is None:
                        break
                    hits.append(float(location.x))
                    step = float(distance) + epsilon
                    origin = location + direction * epsilon
                    travelled += step
                if hits:
                    hits = np.asarray(sorted(hits))
                    # Faces sharing a vertex can report the same crossing more
                    # than once.  Merge only near-identical positions.
                    hits = hits[np.r_[True, np.diff(hits) > epsilon * 4.0]]
                    if len(hits) % 2:
                        # A grazing ray is not a reliable parity line.  Leaving
                        # it empty avoids fabricating a bridge through the shell.
                        continue
                    centers = self.grid_origin[0] + (np.arange(nx) + 0.5) * self.cell_size
                    for lo, hi in hits.reshape(-1, 2):
                        inside[:, iy, iz] |= (centers > lo + epsilon) & (centers < hi - epsilon)
            if iy % max(1, ny // 24) == 0:
                self._progress(0.06 + 0.62 * (iy + 1) / ny)
        if not np.any(inside):
            raise ValueError('No interior voxels were found; increase Volumetric Resolution or use a thicker closed mesh.')
        self._inside = inside
        self._flat_to_node = np.full(total, -1, dtype=np.int32)
        inside_flat = np.flatnonzero(inside.ravel())
        self._flat_to_node[inside_flat] = np.arange(len(inside_flat), dtype=np.int32)
        grid_indices = np.column_stack(np.unravel_index(inside_flat, self.grid_shape)).astype(np.int32)
        self._node_cells = grid_indices
        self._node_positions = self.grid_origin + (grid_indices + 0.5) * self.cell_size
        self.stats['voxel_count'] = int(len(grid_indices))
        self._neighbours = []
        for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
            if dx == dy == dz == 0:
                continue
            self._neighbours.append((dx, dy, dz, self.cell_size * float(np.sqrt(dx * dx + dy * dy + dz * dz))))
        self._node_components = self._components()
        self._vertex_nodes = self._nearest_nodes(self.vertices)
        self.components = self._node_components[self._vertex_nodes]
        self.stats['component_count'] = int(np.max(self._node_components) + 1)
        edges = _surface_edges(self.triangles)
        lengths = np.linalg.norm(self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], axis=1)
        positive = lengths[lengths > 0]
        self.mean_edge_length = float(np.mean(positive)) if len(positive) else self.cell_size

    def _components(self):
        labels = np.full(len(self._node_cells), -1, dtype=np.int32)
        nx, ny, nz = self.grid_shape
        for seed in range(len(labels)):
            if labels[seed] >= 0:
                continue
            component = int(np.max(labels) + 1)
            labels[seed] = component
            stack = [seed]
            while stack:
                node = stack.pop()
                x, y, z = self._node_cells[node]
                for dx, dy, dz, _cost in self._neighbours:
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if not (0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz):
                        continue
                    other = int(self._flat_to_node[np.ravel_multi_index((xx, yy, zz), self.grid_shape)])
                    if other >= 0 and labels[other] < 0:
                        labels[other] = component
                        stack.append(other)
        return labels

    def _nearest_nodes(self, points):
        """Map boundary anchors to nearby interior cells without an O(VN) scan."""
        points = np.asarray(points, dtype=np.float64)
        result = np.empty(len(points), dtype=np.int32)
        nearest_cells = np.rint((points - self.grid_origin) / self.cell_size - 0.5).astype(np.int32)
        nearest_cells = np.clip(nearest_cells, 0, np.asarray(self.grid_shape) - 1)
        unresolved = []
        for point_id, cell in enumerate(nearest_cells):
            node = int(self._flat_to_node[np.ravel_multi_index(tuple(cell), self.grid_shape)])
            if node >= 0:
                result[point_id] = node
            else:
                unresolved.append(point_id)
        # Surface vertices are normally one cell from the interior.  Search
        # successively larger Chebyshev shells so thin features do not force a
        # global nearest-neighbour calculation for every mesh vertex.
        for point_id in unresolved:
            point = points[point_id]
            center = nearest_cells[point_id]
            found = -1
            for radius in range(1, max(self.grid_shape)):
                best_distance = np.inf
                for dx, dy, dz in itertools.product(range(-radius, radius + 1), repeat=3):
                    if max(abs(dx), abs(dy), abs(dz)) != radius:
                        continue
                    cell = center + (dx, dy, dz)
                    if np.any(cell < 0) or np.any(cell >= self.grid_shape):
                        continue
                    node = int(self._flat_to_node[np.ravel_multi_index(tuple(cell), self.grid_shape)])
                    if node < 0:
                        continue
                    position = self.grid_origin + (cell + 0.5) * self.cell_size
                    distance = float(np.dot(point - position, point - position))
                    if distance < best_distance:
                        best_distance, found = distance, node
                if found >= 0:
                    break
            if found < 0:
                raise ValueError('Could not map a mesh anchor to its interior voxel grid.')
            result[point_id] = found
        return result

    def _source(self, indices, weights):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.vertex_count):
            raise ValueError('source index is outside the mesh')
        if weights is None:
            weights = np.ones(len(indices), dtype=np.float64)
            barycentric = False
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if (len(weights) != len(indices) or not np.all(np.isfinite(weights))
                    or np.any(weights < 0) or not np.any(weights > 0)):
                raise ValueError('source_weights must be finite, nonnegative, and contain a positive value')
            weights /= weights.sum()
            barycentric = True
        return indices, weights, barycentric

    def distances(self, source_indices, source_weights=None):
        started = time.perf_counter()
        indices, weights, barycentric = self._source(source_indices, source_weights)
        result = np.full(self.vertex_count, np.inf, dtype=np.float64)
        if not len(indices):
            return result
        if barycentric:
            source_position = np.sum(self.vertices[indices] * weights[:, None], axis=0)
            source_nodes = self._nearest_nodes(source_position[None, :])
        else:
            source_nodes = np.unique(self._vertex_nodes[indices])
        source_components = np.unique(self._node_components[source_nodes])
        distances = np.full(len(self._node_cells), np.inf, dtype=np.float64)
        queue = []
        for node in source_nodes:
            distances[node] = 0.0
            heapq.heappush(queue, (0.0, int(node)))
        nx, ny, nz = self.grid_shape
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            x, y, z = self._node_cells[node]
            for dx, dy, dz, cost in self._neighbours:
                xx, yy, zz = x + dx, y + dy, z + dz
                if not (0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz):
                    continue
                other = int(self._flat_to_node[np.ravel_multi_index((xx, yy, zz), self.grid_shape)])
                if other < 0:
                    continue
                candidate = distance + cost
                if candidate < distances[other]:
                    distances[other] = candidate
                    heapq.heappush(queue, (candidate, other))
        mapped = distances[self._vertex_nodes]
        source_mask = np.isin(self.components, source_components)
        result[source_mask] = mapped[source_mask]
        # The source lies on the boundary while cells lie just inside it.  Shift
        # the sampled field so its supplied face/vertex anchor has distance zero.
        source_level = float(np.dot(result[indices], weights))
        if np.isfinite(source_level):
            result[source_mask] = np.maximum(result[source_mask] - source_level, 0.0)
        self.stats['solve_count'] += 1
        self.stats['grid_solve_count'] += 1
        self.stats['last_solve_seconds'] = time.perf_counter() - started
        return result


def create_volume_solver(vertices, triangles, resolution=48, progress=None):
    return VolumetricSolver(vertices, triangles, resolution=resolution, progress=progress)
