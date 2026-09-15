"""Reusable surface-distance solvers, independent of Blender.

The primary backend is the heat method of Crane, Weischedel and Wardetzky,
"Geodesics in Heat" (2013): https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
We assemble the piecewise-linear triangle FEM stiffness (cotangent Laplacian)
K = G.T A G and lumped vertex mass M, factor M + t K once, and factor an
anchored K once. A query diffuses an integrated Dirac source, normalizes its
negative face gradient, then integrates that field with the Poisson solve.
The time step is mean-edge-length squared per connected component.

Both sparse LU factorizations and the edge adjacency are session resources;
``distances`` never reconstructs them. Sparse factorization memory/fill grows
with mesh size; callers should solve a collapsed preview proxy for dense meshes.
This is a standard cotangent FEM, not an intrinsic Delaunay/tufted Laplacian;
poor triangles can reduce accuracy. Boundary conditions are natural Neumann.

NumPy is required. SciPy availability is checked at runtime: official Blender
distributions do not guarantee SciPy. No dependency installation is attempted.
Missing SciPy, failed factorization, exhausted heat floating-point range on long
thin meshes, and non-surface (wire/degenerate) components use an explicitly
labelled heap-Dijkstra edge-distance approximation. This
respects topology but has directional tessellation bias. Its adjacency is also
built only once. ``backend`` and ``stats['fallback_reason']`` expose the tradeoff.
"""

from __future__ import annotations

import heapq
import time
from typing import Callable

import numpy as np


def _indices(values, width, name, vertex_count):
    values = np.asarray(values)
    if values.size == 0:
        return np.empty((0, width), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer vertex indices")
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= vertex_count):
        raise ValueError(f"{name} contains an out-of-range vertex index")
    return values


def _unique_edges(triangles, extra_edges=None):
    parts = [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    if extra_edges is not None:
        parts.append(extra_edges)
    edges = np.concatenate(parts, axis=0)
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


class GeodesicSolver:
    """Precomputed solver returned by :func:`create_solver`.

    ``distances(indices, weights)`` represents one barycentric point on a face.
    The caller supplies that face's vertices and nonnegative barycentric weights.
    Omitting weights represents a set of vertex sources instead. The output is
    float64, nonnegative, and infinite outside source-connected components.
    Heat distances are approximations, including near a face-interior source.
    """

    def __init__(self, vertices, triangles, edges=None, prefer_heat=True,
                 progress: Callable[[float], None] | None = None):
        started = time.perf_counter()
        self.vertices = np.array(vertices, dtype=np.float64, copy=True)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or not np.all(np.isfinite(self.vertices))):
            raise ValueError("vertices must be a finite array with shape (N, 3)")
        self.vertex_count = len(self.vertices)
        triangles = _indices(triangles, 3, "triangles", self.vertex_count)
        extra_edges = None if edges is None else _indices(
            edges, 2, "edges", self.vertex_count)
        self.triangles = triangles.copy()
        self.edges = _unique_edges(triangles, extra_edges)
        self.stats = {
            "vertex_count": self.vertex_count,
            "triangle_count": len(triangles),
            "edge_count": len(self.edges),
            "adjacency_builds": 1,
            "factorization_count": 0,
            "solve_count": 0,
            "heat_solve_count": 0,
            "graph_solve_count": 0,
            "fallback_reason": "",
        }
        self._heat_lu = None
        self._poisson_lu = None
        self._heat_component_mask = None
        self._progress = progress or (lambda value: None)
        self._progress(0.05)
        self._build_adjacency()
        self._progress(0.22)
        self.backend = "Dijkstra edge approximation"
        if prefer_heat and self.vertex_count:
            self._try_heat_setup()
        elif not prefer_heat:
            self.stats["fallback_reason"] = "Heat method disabled by caller"
        else:
            self.stats["fallback_reason"] = "Empty mesh"
        if self._heat_lu is None:
            self.backend += ": " + self.stats["fallback_reason"]
        self._setup_backend = self.backend
        self.stats["backend"] = self.backend
        self.stats["setup_seconds"] = time.perf_counter() - started
        self._progress(1.0)
        # Do not retain UI callbacks inside numerical session resources.
        self._progress = None

    def _build_adjacency(self):
        adjacency = [[] for _ in range(self.vertex_count)]
        lengths = np.linalg.norm(
            self.vertices[self.edges[:, 0]] - self.vertices[self.edges[:, 1]], axis=1)
        for (i, j), length in zip(self.edges, lengths):
            adjacency[int(i)].append((int(j), float(length)))
            adjacency[int(j)].append((int(i), float(length)))
        self._adjacency = adjacency
        self._edge_lengths = lengths
        positive = lengths[lengths > 0]
        self.mean_edge_length = float(np.mean(positive)) if positive.size else 1.0
        labels = np.full(self.vertex_count, -1, dtype=np.int32)
        component_vertices = []
        for seed in range(self.vertex_count):
            if labels[seed] >= 0:
                continue
            component_id = len(component_vertices)
            labels[seed] = component_id
            stack = [seed]
            members = []
            while stack:
                vertex = stack.pop()
                members.append(vertex)
                for neighbor, _length in adjacency[vertex]:
                    if labels[neighbor] < 0:
                        labels[neighbor] = component_id
                        stack.append(neighbor)
            component_vertices.append(np.asarray(members, dtype=np.int64))
        self.components = labels
        self._component_vertices = component_vertices
        self.stats["component_count"] = len(component_vertices)

    def _try_heat_setup(self):
        try:
            import scipy.sparse as sparse
            from scipy.sparse.linalg import splu
        except (ImportError, OSError, ValueError) as exc:
            self.stats["fallback_reason"] = f"SciPy unavailable ({exc})"
            return
        if not len(self.triangles):
            self.stats["fallback_reason"] = "No nondegenerate triangle surface"
            return
        points = self.vertices[self.triangles]
        edge01 = points[:, 1] - points[:, 0]
        edge02 = points[:, 2] - points[:, 0]
        normals = np.cross(edge01, edge02)
        twice_area = np.linalg.norm(normals, axis=1)
        scale2 = np.maximum(np.einsum("ij,ij->i", edge01, edge01),
                            np.einsum("ij,ij->i", edge02, edge02))
        valid = twice_area > np.maximum(scale2 * 1e-14, np.finfo(float).tiny)
        self.stats["degenerate_triangle_count"] = int(np.count_nonzero(~valid))
        valid_triangles = self.triangles[valid]
        if not len(valid_triangles):
            self.stats["fallback_reason"] = "No nondegenerate triangle surface"
            return

        # Do not pretend a triangle FEM handles wire bridges. Any component
        # containing edges absent from nondegenerate faces uses its cached graph.
        surface_edges = _unique_edges(valid_triangles)
        graph_keys = self.edges[:, 0] * self.vertex_count + self.edges[:, 1]
        surface_keys = surface_edges[:, 0] * self.vertex_count + surface_edges[:, 1]
        wire_edges = self.edges[~np.isin(graph_keys, surface_keys, assume_unique=True)]
        heat_components = np.zeros(len(self._component_vertices), dtype=bool)
        heat_components[np.unique(self.components[valid_triangles.ravel()])] = True
        if len(wire_edges):
            heat_components[np.unique(self.components[wire_edges.ravel()])] = False
        eligible = valid & heat_components[self.components[self.triangles[:, 0]]]
        if not np.any(eligible):
            self.stats["fallback_reason"] = "Components contain wire/degenerate bridges"
            return
        triangle_ids = self.triangles[eligible]
        heat_vertices = np.flatnonzero(heat_components[self.components])
        global_to_heat = np.full(self.vertex_count, -1, dtype=np.int64)
        global_to_heat[heat_vertices] = np.arange(len(heat_vertices))
        local_triangles = global_to_heat[triangle_ids]
        local_count = len(heat_vertices)
        normal_unit = normals[eligible] / twice_area[eligible, None]
        triangle_points = points[eligible]
        opposite_edges = np.stack((triangle_points[:, 2] - triangle_points[:, 1],
                                   triangle_points[:, 0] - triangle_points[:, 2],
                                   triangle_points[:, 1] - triangle_points[:, 0]), axis=1)
        gradients = np.cross(normal_unit[:, None, :], opposite_edges)
        gradients /= twice_area[eligible, None, None]
        areas = 0.5 * twice_area[eligible]
        mass = np.bincount(local_triangles.ravel(),
                           weights=np.repeat(areas / 3.0, 3), minlength=local_count)
        stiffness_values = np.einsum("fik,fjk,f->fij", gradients, gradients, areas)
        row = np.repeat(local_triangles, 3, axis=1).ravel()
        col = np.tile(local_triangles, (1, 3)).ravel()
        stiffness = sparse.coo_matrix((stiffness_values.ravel(), (row, col)),
                                      shape=(local_count, local_count)).tocsc()
        stiffness.sum_duplicates()
        stiffness.eliminate_zeros()
        self._progress(0.40)

        # Each island has its own characteristic length so disconnected objects
        # at very different scales do not over-diffuse one another.
        edge_components = self.components[self.edges[:, 0]]
        length_sums = np.bincount(edge_components, weights=self._edge_lengths,
                                  minlength=len(heat_components))
        edge_counts = np.bincount(edge_components, minlength=len(heat_components))
        average_lengths = length_sums / np.maximum(edge_counts, 1)
        time_steps = average_lengths[self.components[heat_vertices]] ** 2
        heat_matrix = sparse.diags(mass) + sparse.diags(time_steps) @ stiffness
        anchors = np.asarray([global_to_heat[self._component_vertices[c][0]]
                              for c in np.flatnonzero(heat_components)], dtype=np.int64)
        free = np.ones(local_count, dtype=bool)
        free[anchors] = False
        free_ids = np.flatnonzero(free)
        try:
            self._heat_lu = splu(heat_matrix.tocsc(), permc_spec="MMD_AT_PLUS_A")
            self.stats["factorization_count"] += 1
            self._progress(0.65)
            self._poisson_lu = splu(stiffness[free_ids][:, free_ids].tocsc(),
                                    permc_spec="MMD_AT_PLUS_A")
            self.stats["factorization_count"] += 1
        except (RuntimeError, ValueError, MemoryError) as exc:
            self._heat_lu = None
            self._poisson_lu = None
            self.stats["fallback_reason"] = f"Sparse factorization failed ({exc})"
            return
        self._heat_vertices = heat_vertices
        self._global_to_heat = global_to_heat
        self._heat_triangles = local_triangles
        self._face_gradients = gradients
        self._face_areas = areas
        self._poisson_free = free_ids
        self._heat_component_mask = heat_components
        self.stats["heat_vertex_count"] = local_count
        self.stats["heat_component_count"] = int(np.count_nonzero(heat_components))
        self.stats["graph_component_count"] = int(np.count_nonzero(~heat_components))
        self.stats["factor_nnz"] = int(self._heat_lu.L.nnz + self._heat_lu.U.nnz
                                       + self._poisson_lu.L.nnz + self._poisson_lu.U.nnz)
        self.backend = "Heat method (cotangent FEM, prefactored SciPy LU)"
        graph_components = [c for c in np.flatnonzero(~heat_components)
                            if len(self._component_vertices[c]) > 1]
        if graph_components:
            reason = "Wire/degenerate components use Dijkstra edge approximation"
            self.backend += "; " + reason
            self.stats["fallback_reason"] = reason
        self._progress(0.92)

    def _source(self, indices, weights):
        indices = np.asarray(indices)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            if indices.size == 0 and indices.ndim == 1:
                return np.empty(0, dtype=np.int64), np.empty(0), weights is not None
            raise ValueError("source_indices must be a one-dimensional integer array")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.vertex_count):
            raise ValueError("source index is outside the mesh")
        barycentric = weights is not None
        if weights is None:
            indices = np.unique(indices)
            weights = np.ones(len(indices), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if (weights.shape != indices.shape or not np.all(np.isfinite(weights))
                    or np.any(weights < 0)):
                raise ValueError("source_weights must be finite, nonnegative, and match indices")
            nonzero = weights > 0
            indices, weights = indices[nonzero], weights[nonzero]
            if not len(indices):
                raise ValueError("source_weights must have a positive sum")
            if np.unique(self.components[indices]).size != 1:
                raise ValueError("A barycentric source cannot span disconnected components")
            # Normalize via a maximum first to avoid overflow for finite inputs.
            weights = weights / np.max(weights)
            weights = weights / weights.sum()
        return indices, weights, barycentric

    def _graph_distances(self, indices, weights, barycentric):
        result = np.full(self.vertex_count, np.inf, dtype=np.float64)
        if not len(indices):
            return result
        if barycentric:
            point = np.sum(self.vertices[indices] * weights[:, None], axis=0)
            offsets = np.linalg.norm(self.vertices[indices] - point, axis=1)
        else:
            offsets = np.zeros(len(indices), dtype=np.float64)
        queue = []
        for vertex, offset in zip(indices, offsets):
            vertex = int(vertex)
            if offset < result[vertex]:
                result[vertex] = offset
                heapq.heappush(queue, (float(offset), vertex))
        while queue:
            distance, vertex = heapq.heappop(queue)
            if distance > result[vertex]:
                continue
            for neighbor, length in self._adjacency[vertex]:
                candidate = distance + length
                if candidate < result[neighbor]:
                    result[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        self.stats["graph_solve_count"] += 1
        return result

    def distances(self, source_indices, source_weights=None):
        """Return approximate surface distances; reuse all setup/factorizations.

        For a face-interior heat source, calibrate the potential's additive
        constant with the known distances to its face corners. A linear FEM
        cannot represent the source's distance cone inside one triangle; simply
        subtracting interpolated potential would incorrectly zero its corners.
        A graph fallback instead connects a virtual point to the supplied face
        corners with their in-face Euclidean lengths before running Dijkstra.
        Empty source sets return infinity everywhere.
        """
        started = time.perf_counter()
        indices, weights, barycentric = self._source(source_indices, source_weights)
        self.backend = self._setup_backend
        self.stats["last_query_fallback"] = ""
        self.stats["solve_count"] += 1
        if self._heat_lu is None or not len(indices):
            result = self._graph_distances(indices, weights, barycentric)
        else:
            result = np.full(self.vertex_count, np.inf, dtype=np.float64)
            heat_sources = self._heat_component_mask[self.components[indices]]
            if np.any(~heat_sources):
                result = self._graph_distances(indices[~heat_sources], weights[~heat_sources],
                                               barycentric)
            if np.any(heat_sources):
                heat_indices, heat_weights = indices[heat_sources], weights[heat_sources]
                try:
                    heat_result = self._heat_distances(heat_indices, heat_weights, barycentric)
                    finite = np.isfinite(heat_result)
                    result[finite] = heat_result[finite]
                except (RuntimeError, FloatingPointError, ValueError) as exc:
                    # A query failure never causes refactorization in a mouse move.
                    # Preserve the cached matrices; make this query's fallback visible.
                    self.stats["last_query_fallback"] = str(exc)
                    self.backend += f"; current source uses Dijkstra ({exc})"
                    graph = self._graph_distances(heat_indices, heat_weights, barycentric)
                    finite = np.isfinite(graph)
                    result[finite] = graph[finite]
        self.stats["last_solve_seconds"] = time.perf_counter() - started
        self.stats["backend"] = self.backend
        return result

    def _heat_distances(self, indices, weights, barycentric):
        local_indices = self._global_to_heat[indices]
        rhs = np.bincount(local_indices, weights=weights, minlength=len(self._heat_vertices))
        heat = self._heat_lu.solve(rhs)
        if not np.all(np.isfinite(heat)):
            raise FloatingPointError("Heat solve produced nonfinite values")
        active_components = np.unique(self.components[indices])
        for component in active_components:
            members = self._global_to_heat[self._component_vertices[component]]
            amplitude = np.abs(heat[members])
            maximum = float(np.max(amplitude))
            # Short-time heat decays exponentially with intrinsic distance. On
            # very long thin meshes it can exhaust float64's dynamic range even
            # when the sparse solve is finite and reports success. Integrating
            # those zero/subnormal gradients would silently flatten the far
            # distance field. Check only source-reachable components (unreached
            # islands correctly have zero heat), and leave a numerical margin
            # before subnormal gradient normalization becomes unreliable.
            minimum_reliable = max(maximum * 1.0e-280, np.finfo(float).tiny * 1024.0)
            if maximum == 0.0 or np.any(amplitude <= minimum_reliable):
                raise FloatingPointError(
                    "Heat diffusion underflow exhausted floating-point range "
                    "on a reachable component")
        gradients = np.einsum("fi,fij->fj", heat[self._heat_triangles], self._face_gradients)
        # Normalize without squaring tiny heat gradients: they can be exponentially
        # small far from a source and np.linalg.norm alone would underflow.
        scales = np.max(np.abs(gradients), axis=1)
        nonzero = scales > 0
        unit = np.zeros_like(gradients)
        unit[nonzero] = gradients[nonzero] / scales[nonzero, None]
        lengths = np.linalg.norm(unit[nonzero], axis=1)
        unit[nonzero] /= -lengths[:, None]
        divergence_values = np.einsum("fij,fj,f->fi", self._face_gradients, unit,
                                       self._face_areas)
        poisson_rhs = np.bincount(self._heat_triangles.ravel(),
                                  weights=divergence_values.ravel(),
                                  minlength=len(self._heat_vertices))
        potential = np.zeros(len(self._heat_vertices), dtype=np.float64)
        potential[self._poisson_free] = self._poisson_lu.solve(poisson_rhs[self._poisson_free])
        if not np.all(np.isfinite(potential)):
            raise FloatingPointError("Poisson solve produced nonfinite values")
        output = np.full(self.vertex_count, np.inf, dtype=np.float64)
        for component in active_components:
            members = self._component_vertices[component]
            local_members = self._global_to_heat[members]
            selected = self.components[indices] == component
            source_potential = potential[local_indices[selected]]
            if barycentric:
                point = np.sum(self.vertices[indices] * weights[:, None], axis=0)
                corner_distances = np.linalg.norm(self.vertices[indices] - point, axis=1)
                offset = np.dot(source_potential - corner_distances[selected],
                                weights[selected]) / weights[selected].sum()
            else:
                offset = np.min(source_potential)
            values = np.maximum(potential[local_members] - offset, 0.0)
            if not np.any(values > 0.0) and len(members) > 1:
                raise FloatingPointError("Symmetric/degenerate heat source has no distance gradient")
            output[members] = values
        if not barycentric:
            output[indices] = 0.0
        else:
            # The straight segments from a face-interior point to its corners
            # lie on the mesh and are exact surface geodesics.
            output[indices] = corner_distances
        self.stats["heat_solve_count"] += 1
        return output


def create_solver(vertices, triangles, edges=None, prefer_heat=True, progress=None):
    """Create one reusable solver for a fixed triangulated mesh.

    ``progress`` receives monotonically increasing fractions between 0 and 1.
    Pass original mesh edges to preserve loose edges/degenerate topology. Input
    coordinates should share the metric used for strokes (normally world space).
    """
    return GeodesicSolver(vertices, triangles, edges, prefer_heat, progress)
