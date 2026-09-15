"""Integrated dense/proxy sessions using the real registered add-on.

Run with Blender 5.x:
  blender --background --factory-startup --python-exit-code 1 --python tests/test_dense_session_blender.py

These tests exercise real RNA colors and source solves, without GPU drawing or
synthetic modal mouse events. Dense timings therefore are not viewport FPS.
"""

import json
from pathlib import Path
import sys
import unittest

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warpflow
from warpflow import session


PREFIX = "Warpflow Dense Session Test"


def regular_grid(side):
    x, y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    vertices = np.stack((x.ravel(), y.ravel(), np.zeros(side * side)), axis=1)
    corners = np.arange(side * (side - 1))
    corners = corners[corners % side < side - 1]
    faces = np.stack((corners, corners + 1, corners + side + 1, corners + side), axis=1)
    return vertices, faces


def rgba(mesh):
    values = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
    mesh.color_attributes[session.ATTRIBUTE].data.foreach_get("color", values)
    return values.reshape((-1, 4))


class DenseSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the distributed package's dependency preparation and complete
        # registration path, rather than the isolated test namespace bootstrap.
        warpflow.register()
        import scipy
        print("DENSE_SESSION_RUNTIME", bpy.app.version_string, "SciPy", scipy.__version__, scipy.__file__)

    @classmethod
    def tearDownClass(cls):
        session.end_session()
        warpflow.unregister()

    def setUp(self):
        self.s = None
        self.obj = None
        settings = bpy.context.scene.warpflow
        self.saved_settings = {key: getattr(settings, key) for key in
                               ("proxy_threshold", "proxy_target", "sharpness", "strength")}
        settings.sharpness = 2.0
        settings.strength = 1.0

    def tearDown(self):
        session.end_session()
        if self.s is not None:
            self.s.close()
        for obj in list(bpy.data.objects):
            if obj.name.startswith(PREFIX):
                bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            if mesh.name.startswith(PREFIX) and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for key, value in self.saved_settings.items():
            setattr(bpy.context.scene.warpflow, key, value)

    def fixture(self, side, threshold=1000, target=300, wire_bridge=False):
        vertices, faces = regular_grid(side)
        edges = []
        if wire_bridge:
            count = len(vertices)
            second = vertices.copy()
            second[:, 0] += 1.5
            vertices = np.concatenate((vertices, second))
            faces = np.concatenate((faces, faces + count))
            # Connect two otherwise independent face patches by one edge that
            # belongs to no polygon. Decimating only faces would lose the bridge.
            edges = [(side - 1, count)]
        mesh = bpy.data.meshes.new(PREFIX + " Mesh")
        mesh.from_pydata(vertices.tolist(), edges, faces.tolist())
        mesh.update()
        uv = mesh.uv_layers.new(name="UV")
        loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        coordinates = vertices[loop_vertices, :2].copy()
        coordinates[:, 0] /= np.max(vertices[:, 0])
        uv.uv.foreach_set("vector", coordinates.astype(np.float32).ravel())
        obj = bpy.data.objects.new(PREFIX + " Object", mesh)
        bpy.context.scene.collection.objects.link(obj)
        for other in bpy.context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        self.obj = obj
        settings = bpy.context.scene.warpflow
        settings.proxy_threshold = threshold
        settings.proxy_target = target
        self.s = session.PaintSession(bpy.context)
        session.ACTIVE = self.s
        return self.s

    def begin(self, triangle, direction, strength=1.0):
        self.s.settings.strength = strength
        barycentric = np.asarray([0.2, 0.3, 0.5])
        point = np.sum(self.s.vertices[self.s.triangles[triangle]] * barycentric[:, None], axis=0)
        self.s.begin_stroke(triangle, point, (20, 20))
        self.s.drag["direction"] = np.asarray(direction, dtype=np.float64)
        self.s.drag["mouse_end"] = (70, 35)
        self.s.drag["pending"] = True

    def test_proxy_preview_then_full_commit_and_cancel_without_refactorization(self):
        s = self.fixture(45)
        self.assertIsNotNone(s.proxy)
        self.assertTrue(s.proxy.is_proxy)
        self.assertLess(len(s.proxy.vertices), len(s.vertices))
        self.assertIn("Heat method", s.solver.backend)
        self.assertIn("Heat method", s.proxy_solver.backend)
        factors = (s.solver.stats["factorization_count"], s.proxy_solver.stats["factorization_count"])
        self.assertEqual(factors, (2, 2))

        self.begin(0, [1, 0])
        self.assertEqual(len(s.drag["distance"]), len(s.proxy.vertices))
        s.preview()
        self.assertTrue(s.commit_stroke())
        baseline = rgba(self.obj.data).copy()
        np.testing.assert_allclose(session.read_field(self.obj.data), np.tile([1, 0], (len(s.vertices), 1)))

        full_solves = s.solver.stats["heat_solve_count"]
        self.begin(len(s.triangles) - 13, [0, 1], 0.65)
        self.assertEqual(s.solver.stats["heat_solve_count"], full_solves)
        proxy_solves = s.proxy_solver.stats["heat_solve_count"]
        expected_proxy = s.proxy.upsample(s.proxy_accumulator.preview(s.drag["distance"], [0, 1], 0.65))
        for _ in range(3):
            s.preview()
        np.testing.assert_allclose(session.read_field(self.obj.data), expected_proxy, atol=1.3e-7)
        self.assertEqual(s.proxy_solver.stats["heat_solve_count"], proxy_solves)
        self.assertEqual(s.solver.stats["heat_solve_count"], full_solves)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 1)

        # Calculate the original-mesh result independently of the preview route.
        # Direct solver access does not populate the session's distance cache.
        full_distance = s.solver.distances(s.drag["vertices"], s.drag["barycentric"])
        expected_full = s.accumulator.preview(full_distance, [0, 1], 0.65)
        self.assertGreater(float(np.max(np.abs(expected_full - expected_proxy))), 1e-5)
        solves_before_commit = s.solver.stats["heat_solve_count"]
        self.assertTrue(s.commit_stroke())
        self.assertEqual(s.solver.stats["heat_solve_count"], solves_before_commit + 1)
        np.testing.assert_allclose(s.accumulator.field, expected_full, atol=1e-12)
        np.testing.assert_allclose(session.read_field(self.obj.data), expected_full, atol=1.3e-7)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 2)
        committed = rgba(self.obj.data).copy()
        self.assertFalse(np.array_equal(committed, baseline))

        self.begin(len(s.triangles) // 2, [-1, 0])
        s.preview()
        self.assertFalse(np.array_equal(rgba(self.obj.data), committed))
        s.cancel_stroke()
        np.testing.assert_array_equal(rgba(self.obj.data), committed)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 2)
        self.assertEqual((s.solver.stats["factorization_count"], s.proxy_solver.stats["factorization_count"]), factors)
        np.testing.assert_allclose(np.linalg.norm(session.read_field(self.obj.data), axis=1), 1.0, atol=2e-7)

    def test_42025_vertex_default_threshold_smoke(self):
        s = self.fixture(205, threshold=40000, target=5000)
        self.assertGreater(len(s.vertices), 41000)
        self.assertIsNotNone(s.proxy)
        self.assertLess(len(s.proxy.vertices), 10000)
        self.assertIn("Heat method", s.solver.backend)
        factors = (s.solver.stats["factorization_count"], s.proxy_solver.stats["factorization_count"])
        self.begin(30, [1, 0])
        s.preview()
        self.assertTrue(s.commit_stroke())
        self.begin(len(s.triangles) - 50, [0, 1], 0.4)
        s.preview()
        distances = s.solver.distances(s.drag["vertices"], s.drag["barycentric"])
        expected = s.accumulator.preview(distances, [0, 1], 0.4)
        self.assertTrue(s.commit_stroke())
        np.testing.assert_allclose(s.accumulator.field, expected, atol=1e-12)
        np.testing.assert_allclose(session.read_field(self.obj.data), expected, atol=1.3e-7)
        self.assertEqual((s.solver.stats["factorization_count"], s.proxy_solver.stats["factorization_count"]), factors)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 2)
        print("DENSE_SESSION_SMOKE " + json.dumps({
            "vertices": len(s.vertices), "proxy_vertices": len(s.proxy.vertices),
            "session_setup_seconds": s.setup_seconds, "entry_full_solve_ms": s.solve_ms,
            "last_preview_with_attribute_write_ms": s.last_preview_ms,
            "full_factorizations": factors[0], "proxy_factorizations": factors[1],
            "includes_gpu_drawing": False,
        }))

    def test_wire_bridge_preserves_full_connectivity_instead_of_decimating_faces(self):
        s = self.fixture(24, threshold=1000, target=200, wire_bridge=True)
        self.assertGreater(len(s.vertices), s.settings.proxy_threshold)
        self.assertIsNone(s.proxy)
        self.assertIsNone(s.proxy_solver)
        self.assertIsNone(s.proxy_accumulator)
        self.assertIn("Wire edges", s.notice)
        self.assertEqual(len(np.unique(s.solver.components)), 1)
        self.assertIn("Dijkstra", s.solver.backend)
        self.begin(0, [1, 0])
        self.assertEqual(len(s.drag["distance"]), len(s.vertices))
        self.assertTrue(np.all(np.isfinite(s.drag["distance"])))
        first_patch_count = len(s.vertices) // 2
        self.assertGreater(float(np.min(s.drag["distance"][first_patch_count:])), 0.5)
        s.preview()
        self.assertTrue(s.commit_stroke())
        np.testing.assert_allclose(session.read_field(self.obj.data), np.tile([1, 0], (len(s.vertices), 1)))
        committed = rgba(self.obj.data).copy()
        self.begin(len(s.triangles) - 1, [0, 1])
        self.assertTrue(np.all(np.isfinite(s.drag["distance"])))
        s.preview()
        self.assertFalse(np.array_equal(rgba(self.obj.data), committed))
        s.cancel_stroke()
        np.testing.assert_array_equal(rgba(self.obj.data), committed)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise SystemExit(1)
