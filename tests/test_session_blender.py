"""Real Blender test for session transactions, saved properties, and undo.

Run: blender --background --factory-startup --python-exit-code 1 --python tests/test_session_blender.py
The namespace bootstrap avoids importing UI registration during isolated tests.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_warpflow_session_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "warpflow")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / "warpflow" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dependencies = load("dependencies")
dependencies.prepare()
properties = load("properties")
session = load("session")
operators = load("operators")
properties.register()


def colors(mesh):
    values = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
    mesh.color_attributes["Warpflow"].data.foreach_get("color", values)
    return values.reshape((-1, 4))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="warpflow-session-test-")
        mesh = bpy.data.meshes.new("Warpflow Session Test Mesh")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
        mesh.update()
        uv = mesh.uv_layers.new(name="UV")
        uv.uv.foreach_set("vector", np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32).ravel())
        obj = bpy.data.objects.new("Warpflow Session Test Object", mesh)
        bpy.context.scene.collection.objects.link(obj)
        for other in bpy.context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        self.object_name = obj.name
        self.mesh_name = mesh.name
        self.s = None

    @property
    def obj(self):
        return bpy.data.objects.get(self.object_name)

    def tearDown(self):
        session.end_session()
        if self.s is not None:
            self.s.close()
        for obj in list(bpy.data.objects):
            if obj.name.startswith("Warpflow Session Test"):
                bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            if mesh.name.startswith("Warpflow Session Test") and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        self.directory.cleanup()

    def prepare(self):
        self.s = session.PaintSession(bpy.context)
        session.ACTIVE = self.s
        return self.s

    def stroke(self, direction=(1, 0), strength=1):
        self.s.settings.strength = strength
        self.s.begin_stroke(0, self.s.vertices[self.s.triangles[0]].mean(axis=0), (20, 20))
        self.s.drag["direction"] = np.asarray(direction, dtype=np.float64)
        self.s.drag["mouse_end"] = (60, 20)
        self.s.preview()
        return self.s.commit_stroke()

    def test_prepare_is_neutral_and_first_stroke_covers_mesh(self):
        self.prepare()
        np.testing.assert_array_equal(colors(self.obj.data), np.tile([0.5, 0.5, 0, 1], (4, 1)))
        self.assertEqual(len(self.obj.data.warpflow.strokes), 0)
        initial_factors = self.s.solver.stats["factorization_count"]
        self.assertTrue(self.stroke())
        np.testing.assert_array_equal(session.read_field(self.obj.data), np.tile([1, 0], (4, 1)))
        self.assertEqual(len(self.obj.data.warpflow.strokes), 1)
        self.assertEqual(self.s.solver.stats["factorization_count"], initial_factors)

    def test_live_preview_cancel_and_zero_strength_are_transactional(self):
        self.prepare()
        self.stroke()
        baseline = colors(self.obj.data).copy()
        self.s.begin_stroke(0, self.s.vertices[self.s.triangles[0]].mean(axis=0), (20, 20))
        self.s.drag["direction"] = np.array((0, 1))
        self.s.preview()
        self.assertFalse(np.array_equal(colors(self.obj.data), baseline))
        self.assertEqual(len(self.obj.data.warpflow.strokes), 1)
        self.s.cancel_stroke()
        np.testing.assert_array_equal(colors(self.obj.data), baseline)
        self.assertFalse(self.stroke((0, 1), strength=0))
        np.testing.assert_array_equal(colors(self.obj.data), baseline)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 1)

    def test_replay_fractional_strength_matches_saved_color_field(self):
        self.prepare()
        self.stroke((1, 0))
        self.stroke((0, 1), strength=0.35)
        original = colors(self.obj.data).copy()
        session.write_field(self.obj.data, np.zeros((4, 2)))
        self.s.rebuild()
        np.testing.assert_allclose(colors(self.obj.data), original, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(session.read_field(self.obj.data), axis=1), 1, atol=2e-7)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 2)

    def test_property_and_color_persistence_in_blend(self):
        self.prepare()
        self.stroke((0, 1), strength=0.4)
        expected = colors(self.obj.data).copy()
        filename = str(Path(self.directory.name) / "saved.blend")
        bpy.data.libraries.write(filename, {self.obj})
        with bpy.data.libraries.load(filename, link=False) as (source, target):
            target.objects = [self.object_name]
        restored = target.objects[0]
        self.assertIsNot(restored, self.obj)
        self.assertEqual(len(restored.data.warpflow.strokes), 1)
        self.assertAlmostEqual(restored.data.warpflow.strokes[0].strength, 0.4, places=6)
        np.testing.assert_allclose(restored.data.warpflow.strokes[0].direction, [0, 1])
        np.testing.assert_allclose(colors(restored.data), expected)
        self.assertEqual(restored.data.warpflow.geometry_signature, self.s.signature)

    def test_geometry_signature_rebind_and_stale_mesh_detection(self):
        self.prepare()
        self.stroke()
        self.assertTrue(self.s.rebind())
        self.obj.data.vertices[0].co.x += 0.2
        self.obj.data.update()
        self.assertFalse(self.s.rebind())
        with self.assertRaisesRegex(ValueError, "Geometry, transform or UVs changed"):
            session.PaintSession(bpy.context)

    def test_native_undo_redo_rebinds_rna_and_rebuilds_colors(self):
        self.prepare()
        bpy.context.preferences.edit.use_global_undo = True
        bpy.app.handlers.undo_post.append(session.undo_redo_post)
        bpy.app.handlers.redo_post.append(session.undo_redo_post)
        try:
            bpy.ops.ed.undo_push(message="Warpflow Test Start")
            self.stroke((1, 0))
            bpy.ops.ed.undo_push(message="Warpflow Test Stroke 1")
            first = colors(self.obj.data).copy()
            self.stroke((0, 1))
            bpy.ops.ed.undo_push(message="Warpflow Test Stroke 2")
            second = colors(self.obj.data).copy()
            if not bpy.ops.ed.undo.poll():
                self.skipTest("Blender background context does not expose native undo")
            result = bpy.ops.ed.undo()
            self.assertIn("FINISHED", result)
            self.assertIs(session.ACTIVE, self.s)
            self.assertEqual(len(self.obj.data.warpflow.strokes), 1)
            np.testing.assert_allclose(colors(self.obj.data), first, atol=1e-7)
            result = bpy.ops.ed.redo()
            self.assertIn("FINISHED", result)
            self.assertIs(session.ACTIVE, self.s)
            self.assertEqual(len(self.obj.data.warpflow.strokes), 2)
            np.testing.assert_allclose(colors(self.obj.data), second, atol=1e-7)
        finally:
            if session.undo_redo_post in bpy.app.handlers.undo_post:
                bpy.app.handlers.undo_post.remove(session.undo_redo_post)
            if session.undo_redo_post in bpy.app.handlers.redo_post:
                bpy.app.handlers.redo_post.remove(session.undo_redo_post)

    def test_viewport_preview_uses_current_api_and_restores_shading(self):
        self.prepare()
        window = bpy.context.window
        area = next((area for area in window.screen.areas if area.type == "VIEW_3D"), None) if window else None
        if area is None:
            self.skipTest("Background Blender has no View3D area")
        self.s.area = area
        shading = area.spaces.active.shading
        original = (shading.type, shading.color_type, shading.light)
        try:
            shading.type, shading.color_type, shading.light = "WIREFRAME", "OBJECT", "STUDIO"
            self.s.settings.show_preview = True
            self.s.update_preview_shading()
            self.assertEqual((shading.type, shading.color_type, shading.light), ("SOLID", "VERTEX", "FLAT"))
            self.s.settings.show_preview = False
            self.s.update_preview_shading()
            self.assertEqual((shading.type, shading.color_type, shading.light), ("WIREFRAME", "OBJECT", "STUDIO"))
            self.assertIsNone(self.s.preview_saved)
        finally:
            self.s.restore_shading()
            shading.type, shading.color_type, shading.light = original

    def test_drag_back_to_start_restores_field_and_does_not_commit(self):
        self.prepare()
        self.stroke((1, 0))
        baseline = colors(self.obj.data).copy()
        area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
        self.s.area = area
        self.s.begin_stroke(0, self.s.vertices[self.s.triangles[0]].mean(axis=0), (20, 20))
        self.s.drag["direction"] = np.array((0, 1))
        self.s.drag["pending"] = True
        self.s.preview()
        self.assertFalse(np.array_equal(colors(self.obj.data), baseline))
        controller = types.SimpleNamespace(paint_session=self.s, _mouse=lambda event: (21, 20))
        operators.WARPFLOW_OT_paint._update_drag(controller, None)
        self.assertIsNotNone(self.s.drag)
        self.assertIsNone(self.s.drag["direction"])
        self.assertFalse(self.s.drag["pending"])
        np.testing.assert_array_equal(colors(self.obj.data), baseline)
        self.assertFalse(self.s.commit_stroke())
        self.assertEqual(len(self.obj.data.warpflow.strokes), 1)

    def test_undo_preview_toggle_synchronizes_viewport_shading(self):
        self.prepare()
        self.s.area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
        shading = self.s.area.spaces.active.shading
        original = (shading.type, shading.color_type, shading.light)
        bpy.context.preferences.edit.use_global_undo = True
        bpy.app.handlers.undo_post.append(session.undo_redo_post)
        bpy.app.handlers.redo_post.append(session.undo_redo_post)
        try:
            shading.type, shading.color_type, shading.light = "WIREFRAME", "OBJECT", "STUDIO"
            self.s.settings.show_preview = True
            self.s.update_preview_shading()
            bpy.ops.ed.undo_push(message="Warpflow Test Preview On")
            self.s.settings.show_preview = False
            self.s.update_preview_shading()
            bpy.ops.ed.undo_push(message="Warpflow Test Preview Off")
            bpy.ops.ed.undo()
            self.assertTrue(self.s.settings.show_preview)
            self.assertEqual((shading.type, shading.color_type, shading.light), ("SOLID", "VERTEX", "FLAT"))
            bpy.ops.ed.redo()
            self.assertFalse(self.s.settings.show_preview)
            self.assertEqual((shading.type, shading.color_type, shading.light), ("WIREFRAME", "OBJECT", "STUDIO"))
        finally:
            bpy.app.handlers.undo_post.remove(session.undo_redo_post)
            bpy.app.handlers.redo_post.remove(session.undo_redo_post)
            self.s.restore_shading()
            shading.type, shading.color_type, shading.light = original

    def test_changing_editor_still_restores_inactive_viewport_shading(self):
        self.prepare()
        area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
        self.s.area = area
        shading = area.spaces.active.shading
        original = (shading.type, shading.color_type, shading.light)
        try:
            shading.type, shading.color_type, shading.light = "WIREFRAME", "OBJECT", "STUDIO"
            self.s.settings.show_preview = True
            self.s.update_preview_shading()
            area.type = "TEXT_EDITOR"
            self.s.close()
            area.type = "VIEW_3D"
            self.assertEqual((area.spaces.active.shading.type, area.spaces.active.shading.color_type,
                              area.spaces.active.shading.light), ("WIREFRAME", "OBJECT", "STUDIO"))
        finally:
            area.type = "VIEW_3D"
            self.s.restore_shading()
            area.spaces.active.shading.type, area.spaces.active.shading.color_type, area.spaces.active.shading.light = original


if __name__ == "__main__":
    print("Testing sessions with Blender", bpy.app.version_string)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    properties.unregister()
    if not result.wasSuccessful():
        raise SystemExit(1)
