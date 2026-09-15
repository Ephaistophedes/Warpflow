"""Numerical tests run without importing bpy: python -m unittest discover -s tests."""

import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "warpflow_geodesic_test", Path(__file__).parents[1] / "warpflow" / "geodesic.py")
geodesic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geodesic)
try:
    import scipy.sparse.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def grid(size=31):
    vertices = np.array([(x, y, 0.0) for y in np.linspace(-1, 1, size)
                         for x in np.linspace(-1, 1, size)])
    triangles = []
    for y in range(size - 1):
        for x in range(size - 1):
            i = y * size + x
            triangles.extend(((i, i + 1, i + size + 1), (i, i + size + 1, i + size)))
    return vertices, np.asarray(triangles, dtype=np.int64)


class TopologyTests(unittest.TestCase):
    def test_graph_preserves_wire_route_and_disconnected_vertices(self):
        vertices = np.array([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)])
        solver = geodesic.create_solver(vertices, [], [(0, 1), (1, 2), (2, 3)],
                                         prefer_heat=False)
        np.testing.assert_allclose(solver.distances([0])[:4], [0, 1, 2, 3])
        self.assertTrue(np.isinf(solver.distances([0])[4]))
        self.assertEqual(solver.components[0], solver.components[3])
        self.assertNotEqual(solver.components[0], solver.components[4])
        self.assertEqual(solver.stats["adjacency_builds"], 1)

    def test_missing_scipy_is_explicit_and_does_not_install_anything(self):
        original_import = builtins.__import__

        def import_without_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ModuleNotFoundError("test environment has no scipy")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_scipy):
            solver = geodesic.create_solver(*grid(3))
        self.assertIn("SciPy unavailable", solver.backend)
        self.assertEqual(solver.stats["factorization_count"], 0)
        self.assertTrue(np.all(np.isfinite(solver.distances([0]))))

    def test_barycentric_graph_source_has_correct_corner_offsets(self):
        vertices = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
        solver = geodesic.create_solver(vertices, [(0, 1, 2)], prefer_heat=False)
        result = solver.distances([0, 1, 2], [0.2, 0.3, 0.5])
        expected = np.linalg.norm(vertices - [0.3, 0.5, 0], axis=1)
        np.testing.assert_allclose(result, expected)

    def test_empty_isolated_and_degenerate_meshes(self):
        empty = geodesic.create_solver(np.empty((0, 3)), [])
        self.assertEqual(empty.distances([]).shape, (0,))
        vertices = np.array([(0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (8., 0., 0.)])
        solver = geodesic.create_solver(vertices, [(0, 1, 2)])
        np.testing.assert_allclose(solver.distances([0])[:3], [0, 1, 2])
        isolated = solver.distances([3])
        self.assertEqual(isolated[3], 0)
        self.assertTrue(np.all(np.isinf(isolated[:3])))
        self.assertTrue(np.all(np.isinf(solver.distances([]))))

    def test_rejects_invalid_sources_and_geometry(self):
        solver = geodesic.create_solver(*grid(3), prefer_heat=False)
        for indices, weights in (([-1], None), ([9], None), ([1.5], None),
                                  ([0, 1], [0, 0]), ([0, 1], [-1, 2]),
                                  ([0, 1], [np.nan, 1]), ([0, 1], [1])):
            with self.subTest(indices=indices, weights=weights), self.assertRaises(ValueError):
                solver.distances(indices, weights)
        with self.assertRaises(ValueError):
            geodesic.create_solver([[0, 0, np.nan]], [])
        with self.assertRaises(ValueError):
            geodesic.create_solver([[0, 0, 0]], [(0, 1, 2)])
        disconnected = geodesic.create_solver([[0, 0, 0], [1, 0, 0]], [])
        with self.assertRaisesRegex(ValueError, "disconnected"):
            disconnected.distances([0, 1], [0.5, 0.5])


@unittest.skipUnless(HAS_SCIPY, "SciPy unavailable: heat backend tests require it")
class HeatMethodTests(unittest.TestCase):
    def test_planar_distance_matches_analytic_field(self):
        vertices, triangles = grid(31)
        solver = geodesic.create_solver(vertices, triangles)
        source = len(vertices) // 2
        result = solver.distances([source])
        expected = np.linalg.norm(vertices - vertices[source], axis=1)
        self.assertTrue(solver.backend.startswith("Heat method"))
        self.assertLess(np.mean(np.abs(result - expected)), 0.03)
        self.assertLess(np.max(np.abs(result - expected)), 0.09)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result[source], 0)
        self.assertTrue(np.all(result >= 0))

    def test_factorizations_and_adjacency_reused_for_new_sources(self):
        progress = []
        solver = geodesic.create_solver(*grid(15), progress=progress.append)
        heat_lu, poisson_lu = solver._heat_lu, solver._poisson_lu
        adjacency = solver._adjacency
        with mock.patch("scipy.sparse.linalg.splu", side_effect=AssertionError("refactorized")):
            for source in [0, 60, 125, 224]:
                solver.distances([source])
        self.assertIs(heat_lu, solver._heat_lu)
        self.assertIs(poisson_lu, solver._poisson_lu)
        self.assertIs(adjacency, solver._adjacency)
        self.assertEqual(solver.stats["factorization_count"], 2)
        self.assertEqual(solver.stats["heat_solve_count"], 4)
        self.assertEqual(solver.stats["graph_solve_count"], 0)
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 1.0)

    def test_barycentric_heat_source_corners_are_not_zeroed(self):
        vertices, triangles = grid(17)
        solver = geodesic.create_solver(vertices, triangles)
        face = triangles[231]
        weights = np.array([0.2, 0.35, 0.45])
        point = np.sum(vertices[face] * weights[:, None], axis=0)
        result = solver.distances(face, weights)
        np.testing.assert_allclose(result[face], np.linalg.norm(vertices[face] - point, axis=1))
        self.assertTrue(np.all(result[face] > 0))
        expected = np.linalg.norm(vertices - point, axis=1)
        self.assertLess(np.mean(np.abs(result - expected)), 0.1)
        self.assertEqual(solver.stats["heat_solve_count"], 1)

    def test_disconnected_surface_and_wire_keep_independent_backends(self):
        first, triangles = grid(7)
        vertices = np.concatenate((first, first, [(10, 0, 0), (11, 0, 0), (12, 0, 0)]))
        triangles = np.concatenate((triangles, triangles + len(first)))
        offset = 2 * len(first)
        solver = geodesic.create_solver(vertices, triangles, [(offset, offset + 1)])
        surface = solver.distances([0])
        self.assertTrue(np.all(np.isfinite(surface[:len(first)])))
        self.assertTrue(np.all(np.isinf(surface[len(first):])))
        wire = solver.distances([offset])
        self.assertEqual(wire[offset + 1], 1)
        self.assertTrue(np.isinf(wire[offset + 2]))
        self.assertIn("Heat method", solver.backend)
        self.assertIn("Dijkstra", solver.backend)
        self.assertEqual(solver.stats["factorization_count"], 2)

    def test_geodesic_respects_hole(self):
        vertices, triangles = grid(41)
        centers = vertices[triangles].mean(axis=1)
        keep = ~((np.abs(centers[:, 0]) < 0.30) & (np.abs(centers[:, 1]) < 0.60))
        solver = geodesic.create_solver(vertices, triangles[keep])
        source = int(np.argmin(np.linalg.norm(vertices - [-0.6, 0, 0], axis=1)))
        target = int(np.argmin(np.linalg.norm(vertices - [0.6, 0, 0], axis=1)))
        distance = solver.distances([source])[target]
        self.assertGreater(distance, 1.65)  # Straight-line distance would be 1.2.
        self.assertLess(distance, 2.4)
        self.assertEqual(solver.stats["heat_solve_count"], 1)

    def test_isometric_fold_preserves_surface_distance(self):
        flat, triangles = grid(17)
        folded = flat.copy()
        positive = folded[:, 0] > 0
        folded[positive, 2] = folded[positive, 0]
        folded[positive, 0] = 0
        source = 8 * 17
        flat_result = geodesic.create_solver(flat, triangles).distances([source])
        folded_result = geodesic.create_solver(folded, triangles).distances([source])
        np.testing.assert_allclose(folded_result, flat_result, atol=1e-9, rtol=1e-9)

    def test_real_factorization_failure_falls_back(self):
        with mock.patch("scipy.sparse.linalg.splu", side_effect=RuntimeError("singular test")):
            solver = geodesic.create_solver(*grid(5))
        self.assertIn("Sparse factorization failed", solver.backend)
        self.assertTrue(np.all(np.isfinite(solver.distances([0]))))
        self.assertEqual(solver.stats["heat_solve_count"], 0)

    def test_long_strip_heat_underflow_uses_explicit_query_fallback(self):
        # The diffusion value underflows after roughly 850 edges on this strip.
        # Silently zeroing its distant gradients used to give a ~848-unit
        # distance plateau despite the far endpoint being 2,000 units away.
        length = 2000
        vertices = np.array([(float(x), y, 0.0)
                             for y in (0.0, 1.0) for x in range(length + 1)])
        corners = np.arange(length, dtype=np.int64)
        triangles = np.concatenate((
            np.stack((corners, corners + 1, corners + length + 2), axis=1),
            np.stack((corners, corners + length + 2, corners + length + 1), axis=1),
        ))
        solver = geodesic.create_solver(vertices, triangles)
        heat_lu, poisson_lu = solver._heat_lu, solver._poisson_lu
        result = solver.distances([0])
        np.testing.assert_allclose(result[:length + 1], np.arange(length + 1), atol=1e-9)
        self.assertIn("Dijkstra", solver.backend)
        self.assertIn("underflow", solver.backend)
        self.assertIn("underflow", solver.stats["last_query_fallback"])
        self.assertEqual(solver.stats["graph_solve_count"], 1)
        self.assertEqual(solver.stats["factorization_count"], 2)
        self.assertIs(heat_lu, solver._heat_lu)
        self.assertIs(poisson_lu, solver._poisson_lu)


if __name__ == "__main__":
    unittest.main()
