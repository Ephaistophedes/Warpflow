"""Run: blender --background --factory-startup --python-exit-code 1 --python tests/test_uv_bake_blender.py.

Tests real Blender 5.x RNA, native Cycles baking, raw PNG channels, and cleanup.
Loads only the two modules under test so interactive operator state is irrelevant.
"""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_warpflow_uv_bake_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "warpflow")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / "warpflow" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


uv = load("uv")
bake = load("bake")


def make_mesh(vertices=None, faces=None, uvs=None):
    mesh = bpy.data.meshes.new("Warpflow Test Mesh")
    mesh.from_pydata(vertices or [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], faces or [(0, 1, 2, 3)])
    mesh.update()
    layer = mesh.uv_layers.new(name="Flow UV")
    coordinates = uvs if uvs is not None else [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    layer.uv.foreach_set("vector", np.asarray(coordinates, dtype=np.float32).ravel())
    return mesh


class UVTests(unittest.TestCase):
    def tearDown(self):
        for mesh in list(bpy.data.meshes):
            if mesh.name.startswith("Warpflow Test") and mesh.users == 0:
                bpy.data.meshes.remove(mesh)

    def test_plane_and_projection(self):
        data = uv.build_uv_data(make_mesh())
        self.assertEqual(data.island_count, 1)
        np.testing.assert_allclose(data.vertex_tangents, np.tile([1, 0, 0], (4, 1)), atol=1e-7)
        np.testing.assert_allclose(data.vertex_bitangents, np.tile([0, 1, 0], (4, 1)), atol=1e-7)
        np.testing.assert_allclose(data.direction_to_uv(0, [2, 2, 3]), np.sqrt([0.5, 0.5]))
        self.assertIsNone(data.direction_to_uv(0, [0, 0, 2]))

    def test_mirrored_handedness(self):
        data = uv.build_uv_data(make_mesh(uvs=[(0.9, 0.1), (0.1, 0.1), (0.1, 0.9), (0.9, 0.9)]))
        np.testing.assert_allclose(data.triangle_tangents, np.tile([-1, 0, 0], (2, 1)), atol=1e-7)
        np.testing.assert_allclose(data.triangle_bitangents, np.tile([0, 1, 0], (2, 1)), atol=1e-7)
        np.testing.assert_allclose(data.direction_to_uv(0, [1, 0, 0]), [-1, 0])

    def test_uv_islands_and_shared_vertices(self):
        mesh = make_mesh(vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)],
                         faces=[(0, 1, 2, 3), (1, 4, 5, 2)],
                         uvs=[(0, 0), (0.4, 0), (0.4, 1), (0, 1), (0.6, 0), (1, 0), (1, 1), (0.6, 1)])
        data = uv.build_uv_data(mesh)
        self.assertEqual(data.island_count, 2)
        self.assertTrue(any("2 vertices have split" in warning for warning in data.warnings))
        # Dominance is the incident triangulated surface area at this vertex,
        # which need not tie even when the neighboring quads have equal areas.
        for vertex in (1, 2):
            area = np.zeros(2)
            for triangle in mesh.loop_triangles:
                if vertex in triangle.vertices:
                    area[data.triangle_islands[triangle.index]] += triangle.area
            self.assertEqual(data.vertex_islands[vertex], int(np.argmax(area)))

    def test_curved_island_local_frames(self):
        mesh = make_mesh(vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)],
                         faces=[(0, 1, 2, 3), (1, 4, 5, 2)],
                         uvs=[(0, 0), (0.5, 0), (0.5, 1), (0, 1), (0.5, 0), (1, 0), (1, 1), (0.5, 1)])
        data = uv.build_uv_data(mesh)
        self.assertEqual(data.island_count, 1)
        np.testing.assert_allclose(np.linalg.norm(data.vertex_tangents, axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(data.vertex_bitangents, axis=1), 1)
        np.testing.assert_allclose(np.sum(data.vertex_tangents * data.vertex_bitangents, axis=1), 0, atol=1e-7)
        self.assertGreater(data.vertex_tangents[1, 2], 0.2)

    def test_cut_seam_warns_even_with_one_connected_island(self):
        vertices = [(np.cos(angle), np.sin(angle), z)
                    for z in (0, 1) for angle in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
        faces = [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
        coordinates = [(u, v) for i in range(4)
                       for u, v in ((i / 4, 0), ((i + 1) / 4, 0), ((i + 1) / 4, 1), (i / 4, 1))]
        data = uv.build_uv_data(make_mesh(vertices=vertices, faces=faces, uvs=coordinates))
        self.assertEqual(data.island_count, 1)
        self.assertTrue(any("2 vertices have split" in warning for warning in data.warnings))

    def test_invalid_unwrap(self):
        mesh = make_mesh(uvs=[(0, 0)] * 4)
        with self.assertRaisesRegex(ValueError, "Degenerate UV"):
            uv.build_uv_data(mesh)
        mesh.uv_layers.remove(mesh.uv_layers.active)
        with self.assertRaisesRegex(ValueError, "UV unwrap"):
            uv.build_uv_data(mesh)


class BakeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="warpflow-bake-test-")
        self.mesh = make_mesh()
        self.obj = bpy.data.objects.new("Warpflow Test Object", self.mesh)
        bpy.context.scene.collection.objects.link(self.obj)
        self.attr = self.mesh.color_attributes.new(name="Warpflow", type="FLOAT_COLOR", domain="POINT")
        self.source = np.tile([1, 0.5, 0.25, 0.7], (4, 1)).astype(np.float32)
        self.attr.data.foreach_set("color", self.source.ravel())
        self.material = bpy.data.materials.new("Warpflow Test Artist Material")
        self.material.use_nodes = True
        self.mesh.materials.append(self.material)
        self.obj.modifiers.new("Artist subdivision", "SUBSURF")
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

    def tearDown(self):
        bpy.data.objects.remove(self.obj, do_unlink=True)
        bpy.data.meshes.remove(self.mesh)
        bpy.data.materials.remove(self.material)
        self.directory.cleanup()

    def snapshot(self):
        scene = bpy.context.scene
        return (scene.as_pointer(), bpy.context.view_layer.objects.active.as_pointer(),
                tuple(sorted(o.as_pointer() for o in bpy.context.selected_objects)),
                scene.render.engine, scene.render.bake.margin, scene.render.bake.use_clear,
                scene.view_settings.view_transform, scene.view_settings.look,
                scene.render.image_settings.file_format, scene.render.image_settings.color_depth,
                tuple(m.as_pointer() for m in self.mesh.materials), len(self.obj.modifiers),
                tuple((node.name, node.select) for node in self.material.node_tree.nodes),
                tuple(len(getattr(bpy.data, name)) for name in ("objects", "meshes", "scenes", "materials", "images")))

    def test_roundtrip_raw_rg_and_complete_state_restoration(self):
        before = self.snapshot()
        destination = bake.bake_flowmap(bpy.context, self.obj, str(Path(self.directory.name) / "flow.png"), 32, margin=2)
        self.assertEqual(self.snapshot(), before)
        original = np.empty(16, dtype=np.float32)
        self.attr.data.foreach_get("color", original)
        np.testing.assert_array_equal(original, self.source.ravel())
        image = bpy.data.images.load(destination, check_existing=False)
        try:
            image.colorspace_settings.name = "Non-Color"
            pixels = np.empty(32 * 32 * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            pixels = pixels.reshape((32, 32, 4))
            np.testing.assert_allclose(pixels[16, 16], [1, 0.5, 0, 1], atol=2e-5)
            np.testing.assert_allclose(pixels[0, 0], [0.5, 0.5, 0, 1], atol=2e-5)
        finally:
            bpy.data.images.remove(image)

    def test_save_failure_restores_state_and_preserves_existing_file(self):
        destination = Path(self.directory.name) / "existing.png"
        destination.write_bytes(b"existing artist file")
        before = self.snapshot()
        with patch.object(bake.os, "replace", side_effect=PermissionError("test failure")):
            with self.assertRaisesRegex(PermissionError, "test failure"):
                bake.bake_flowmap(bpy.context, self.obj, str(destination), 16, margin=0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(destination.read_bytes(), b"existing artist file")
        self.assertEqual(sorted(os.listdir(self.directory.name)), ["existing.png"])

    def test_out_of_tile_rejected_before_state_changes(self):
        self.mesh.uv_layers.active.uv[0].vector = (1.2, 0.1)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "one PNG tile"):
            bake.bake_flowmap(bpy.context, self.obj, str(Path(self.directory.name) / "flow.png"), 16)
        self.assertEqual(self.snapshot(), before)

    def test_operator_failure_restores_every_datablock(self):
        before = self.snapshot()
        # bpy.ops dynamically creates a new proxy on each access, so replace the
        # module reference rather than patching one ephemeral operator proxy.
        failing_ops = types.SimpleNamespace(object=types.SimpleNamespace(
            bake=Mock(side_effect=RuntimeError("simulated Cycles failure"))))
        fake_bpy = types.SimpleNamespace(data=bpy.data, path=bpy.path, ops=failing_ops)
        with patch.object(bake, "bpy", fake_bpy):
            with self.assertRaisesRegex(RuntimeError, "simulated Cycles failure"):
                bake.bake_flowmap(bpy.context, self.obj, str(Path(self.directory.name) / "flow.png"), 16)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(os.listdir(self.directory.name), [])

    def test_native_gradient_bake_has_unit_pixel_directions(self):
        colors = np.array([[1, 0.5, 0, 1], [0.5, 1, 0, 1], [0.5, 1, 0, 1], [1, 0.5, 0, 1]], dtype=np.float32)
        self.attr.data.foreach_set("color", colors.ravel())
        destination = bake.bake_flowmap(bpy.context, self.obj, str(Path(self.directory.name) / "gradient.png"), 32, margin=0)
        image = bpy.data.images.load(destination, check_existing=False)
        try:
            image.colorspace_settings.name = "Non-Color"
            pixels = np.empty(32 * 32 * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            center = pixels.reshape((32, 32, 4))[8:24, 8:24]
            np.testing.assert_allclose(np.linalg.norm(center[:, :, :2] * 2 - 1, axis=2), 1, atol=4e-5)
            self.assertGreater(center[8, 8, 0], 0.8)
            self.assertGreater(center[8, 8, 1], 0.8)
        finally:
            bpy.data.images.remove(image)

    def test_neutral_bake_does_not_invent_directions(self):
        self.attr.data.foreach_set("color", np.tile([0.5, 0.5, 0, 1], 4))
        destination = bake.bake_flowmap(bpy.context, self.obj, str(Path(self.directory.name) / "neutral.png"), 16)
        image = bpy.data.images.load(destination, check_existing=False)
        try:
            image.colorspace_settings.name = "Non-Color"
            pixels = np.empty(16 * 16 * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            np.testing.assert_allclose(pixels.reshape((-1, 4)), np.tile([0.5, 0.5, 0, 1], (256, 1)), atol=2e-5)
        finally:
            bpy.data.images.remove(image)

    def test_interpolated_pixels_remain_normalized(self):
        # Pixel math is tested independently of a particular Cycles raster sample.
        pixels = np.array([[0.75, 0.75, 0.2, 0.3], [0.5, 0.5, 1, 0]], dtype=np.float32)
        bake._normalize_pixels(pixels)
        self.assertAlmostEqual(float(np.linalg.norm(pixels[0, :2] * 2 - 1)), 1.0, places=6)
        np.testing.assert_allclose(pixels[1], [0.5, 0.5, 0, 1])


if __name__ == "__main__":
    print("Testing UV/bake with Blender", bpy.app.version_string)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise SystemExit(1)
