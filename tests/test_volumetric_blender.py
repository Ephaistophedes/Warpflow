"""Volumetric distance-mode coverage in Blender's Python runtime.

Run: blender --background --factory-startup --python-exit-code 1 --python tests/test_volumetric_blender.py
"""

from pathlib import Path
import sys
import unittest

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warpflow
from warpflow import session
from warpflow import geodesic


PREFIX = 'Warpflow Volume Test'


def cube_object():
    vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(PREFIX + ' Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name='UV')
    uv.uv.foreach_set('vector', np.tile(np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32), (6, 1)).ravel())
    obj = bpy.data.objects.new(PREFIX + ' Object', mesh)
    bpy.context.scene.collection.objects.link(obj)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


class VolumetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warpflow.register()

    @classmethod
    def tearDownClass(cls):
        session.end_session()
        warpflow.unregister()

    def setUp(self):
        self.obj = cube_object()
        self.settings = bpy.context.scene.warpflow
        self.saved = (self.settings.distance_mode, self.settings.volume_resolution,
                      self.settings.proxy_threshold)
        self.settings.volume_resolution = 32
        self.settings.proxy_threshold = 1
        self.s = None

    def tearDown(self):
        session.end_session()
        if self.s is not None:
            self.s.close()
        self.settings.distance_mode, self.settings.volume_resolution, self.settings.proxy_threshold = self.saved
        mesh = self.obj.data
        bpy.data.objects.remove(self.obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    def prepare(self, mode):
        self.settings.distance_mode = mode
        self.s = session.PaintSession(bpy.context)
        return self.s

    def test_volume_mode_uses_interior_and_avoids_surface_proxy(self):
        volume = self.prepare('VOLUME')
        self.assertEqual(volume.distance_mode, 'VOLUME')
        self.assertIn('Volumetric voxel', volume.solver.backend)
        self.assertIsNone(volume.proxy_solver)
        self.assertIn('interior voxel', volume.notice)
        source_triangle = next(tri for tri in volume.triangles if np.all(volume.vertices[tri, 0] > 0.9))
        opposite_triangle = next(tri for tri in volume.triangles if np.all(volume.vertices[tri, 0] < -0.9))
        weights = np.full(3, 1.0 / 3.0)
        volume_distance = volume.solver.distances(source_triangle, weights)
        self.assertTrue(np.all(np.isfinite(volume_distance[opposite_triangle])))
        # A path from opposite cube faces crosses the solid directly, rather
        # than travelling around four surface edges.
        self.assertLess(float(np.min(volume_distance[opposite_triangle])), 3.0)
        volume.close()
        self.s = None

        # Compare to an edge-graph surface path.  The heat method deliberately
        # smooths distance on this coarse six-quad cube, while the graph gives
        # the unambiguous surface-only route for this topology assertion.
        surface_distance = geodesic.create_solver(volume.vertices, volume.triangles,
                                                   volume.edges, prefer_heat=False).distances(source_triangle, weights)
        self.assertGreater(float(np.min(surface_distance[opposite_triangle])),
                           float(np.min(volume_distance[opposite_triangle])) + 0.3)

    def test_volume_mode_rejects_open_meshes(self):
        bpy.data.objects.remove(self.obj, do_unlink=True)
        mesh = bpy.data.meshes.new(PREFIX + ' Open Mesh')
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        mesh.update()
        uv = mesh.uv_layers.new(name='UV')
        uv.uv.foreach_set('vector', np.array(((0, 0), (1, 0), (0, 1)), dtype=np.float32).ravel())
        self.obj = bpy.data.objects.new(PREFIX + ' Open Object', mesh)
        bpy.context.scene.collection.objects.link(self.obj)
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj
        self.settings.distance_mode = 'VOLUME'
        with self.assertRaisesRegex(ValueError, 'closed, manifold'):
            session.PaintSession(bpy.context)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise SystemExit(1)
