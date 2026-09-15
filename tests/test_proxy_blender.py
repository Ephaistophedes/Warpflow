"""Run with Blender --background --factory-startup --python this_file.py."""

from pathlib import Path
import sys
import time
import unittest

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warpflow.proxy import build_proxy, _topology_signature, _continuous_transfer


def grid(side=48, hole=False):
    vertices = np.asarray([(x / (side - 1), y / (side - 1), 0.0)
                           for y in range(side) for x in range(side)])
    triangles = []
    for y in range(side - 1):
        for x in range(side - 1):
            if hole and side // 3 <= x < 2 * side // 3 and side // 3 <= y < 2 * side // 3:
                continue
            a = y * side + x
            triangles.extend(((a, a + 1, a + side + 1), (a, a + side + 1, a + side)))
    triangles = np.asarray(triangles, dtype=np.int32)
    used = np.unique(triangles)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[triangles]


class ProxyTests(unittest.TestCase):
    def test_grid_collapses_and_transfers_unit_field(self):
        vertices, triangles = grid(70)
        before_objects = len(bpy.data.objects)
        before_meshes = len(bpy.data.meshes)
        started = time.perf_counter()
        proxy = build_proxy(None, vertices, triangles, np.zeros(len(vertices), dtype=np.int32), target_vertices=700)
        print("PROXY_BENCHMARK", proxy.stats, "total", time.perf_counter() - started)
        self.assertTrue(proxy.is_proxy, proxy.stats)
        self.assertLess(len(proxy.vertices), len(vertices) // 2)
        self.assertEqual(_topology_signature(len(vertices), triangles),
                         _topology_signature(len(proxy.vertices), proxy.triangles))
        preview = proxy.upsample(np.tile([0.6, 0.8], (len(proxy.vertices), 1)))
        np.testing.assert_allclose(preview, np.tile([0.6, 0.8], (len(vertices), 1)), atol=1e-10)
        self.assertEqual(len(bpy.data.objects), before_objects)
        self.assertEqual(len(bpy.data.meshes), before_meshes)
        source, weights = proxy.map_source(triangles[100], [0.2, 0.3, 0.5])
        self.assertTrue(np.all(source >= 0))
        self.assertTrue(np.all(source < len(proxy.vertices)))
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertTrue(np.all(weights >= 0))

    def test_holes_and_coincident_disconnected_components_stay_separate(self):
        local_vertices, local_triangles = grid(42, hole=True)
        vertices = np.concatenate((local_vertices, local_vertices))
        triangles = np.concatenate((local_triangles, local_triangles + len(local_vertices)))
        components = np.repeat([0, 1], len(local_vertices))
        proxy = build_proxy(None, vertices, triangles, components, target_vertices=700)
        print("HOLE_PROXY", proxy.stats)
        self.assertTrue(proxy.is_proxy, proxy.stats)
        for label in (0, 1):
            p_indices = np.flatnonzero(proxy.components == label)
            p_triangles = proxy.triangles[proxy.components[proxy.triangles[:, 0]] == label] - p_indices[0]
            self.assertEqual(_topology_signature(len(local_vertices), local_triangles),
                             _topology_signature(len(p_indices), p_triangles))
            source_full = local_triangles[20] + label * len(local_vertices)
            source, weights = proxy.map_source(source_full, [0.1, 0.2, 0.7])
            np.testing.assert_array_equal(proxy.components[source], label)
        field = np.zeros((len(proxy.vertices), 2))
        field[proxy.components == 0] = [1, 0]
        field[proxy.components == 1] = [0, 1]
        full = proxy.upsample(field)
        np.testing.assert_array_equal(full[components == 0], np.tile([1, 0], (len(local_vertices), 1)))
        np.testing.assert_array_equal(full[components == 1], np.tile([0, 1], (len(local_vertices), 1)))

    def test_nonmanifold_and_loose_vertices_use_exact_fallback(self):
        vertices, triangles = grid(25)
        # An isolated vertex in a declared component makes it unsafe to collapse.
        vertices = np.concatenate((vertices, [[2, 2, 2]]))
        proxy = build_proxy(None, vertices, triangles, np.zeros(len(vertices), dtype=np.int32), target_vertices=50)
        self.assertFalse(proxy.is_proxy)
        self.assertTrue(proxy.stats["fallbacks"])
        source, weights = proxy.map_source([len(vertices) - 1], [1.0])
        np.testing.assert_array_equal(source, [len(vertices) - 1])

    def test_curved_surface_reduces_without_crossing_folds(self):
        vertices, triangles = grid(60)
        angle = vertices[:, 0] * np.pi
        vertices[:, 0] = np.cos(angle)
        vertices[:, 2] = np.sin(angle)
        proxy = build_proxy(None, vertices, triangles, np.zeros(len(vertices), dtype=np.int32), target_vertices=600)
        self.assertTrue(proxy.is_proxy, proxy.stats)
        self.assertFalse(proxy.stats["fallbacks"])
        self.assertEqual(_topology_signature(len(vertices), triangles),
                         _topology_signature(len(proxy.vertices), proxy.triangles))

    def test_transfer_rejects_topological_jump(self):
        full_triangles = np.asarray([[0, 1, 2]])
        proxy_triangles = np.asarray([[0, 1, 2], [3, 4, 5]])
        self.assertFalse(_continuous_transfer(full_triangles, proxy_triangles, np.asarray([0, 0, 1])))


if __name__ == "__main__":
    result = unittest.main(argv=[sys.argv[0]], exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
