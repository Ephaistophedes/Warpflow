"""Real Blender tests for persistent, editable stroke arrow constraints.

Run: blender --background --factory-startup --python-exit-code 1 --python tests/test_stroke_edit_blender.py
Uses the actual registered addon and bundled SciPy, including native undo hooks.
"""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import bpy
import numpy as np
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warpflow
from warpflow import operators, session

warpflow.register()


def colors(mesh):
    values = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
    mesh.color_attributes["Warpflow"].data.foreach_get("color", values)
    return values.reshape((-1, 4))


def saved_records(mesh):
    """Plain immutable snapshots remain valid after Blender replaces RNA."""
    return [
        (tuple(stroke.vertices), tuple(stroke.barycentric), tuple(stroke.direction),
         stroke.strength, stroke.has_tip, tuple(stroke.tip_vertices),
         tuple(stroke.tip_barycentric))
        for stroke in mesh.warpflow.strokes
    ]


class StrokeEditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="warpflow-stroke-edit-")
        side = 6
        points = [(x / (side - 1), y / (side - 1), 0.0)
                  for y in range(side) for x in range(side)]
        triangles = []
        for y in range(side - 1):
            for x in range(side - 1):
                a = y * side + x
                triangles.extend(((a, a + 1, a + side + 1),
                                  (a, a + side + 1, a + side)))
        # A separate surface catches accidental endpoint jumps across objects'
        # disconnected components without making the normal fixture expensive.
        island = len(points)
        points.extend(((3, 0, 0), (4, 0, 0), (3, 1, 0)))
        triangles.append((island, island + 1, island + 2))
        mesh = bpy.data.meshes.new("Warpflow Stroke Edit Test Mesh")
        mesh.from_pydata(points, [], triangles)
        mesh.update()
        uv = mesh.uv_layers.new(name="UV")
        uv.uv.foreach_set("vector", np.asarray(
            [points[loop.vertex_index][:2] for loop in mesh.loops],
            dtype=np.float32).ravel())
        obj = bpy.data.objects.new("Warpflow Stroke Edit Test Object", mesh)
        bpy.context.scene.collection.objects.link(obj)
        for other in bpy.context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        self.object_name = obj.name
        self.s = None
        bpy.context.scene.warpflow.strength = 1
        bpy.context.scene.warpflow.sharpness = 2
        bpy.context.scene.warpflow.proxy_threshold = 40000

    @property
    def obj(self):
        return bpy.data.objects.get(self.object_name)

    def tearDown(self):
        session.end_session()
        if self.s is not None:
            self.s.close()
        for obj in list(bpy.data.objects):
            if obj.name.startswith("Warpflow Stroke Edit Test"):
                bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            if mesh.name.startswith("Warpflow Stroke Edit Test") and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        self.directory.cleanup()

    def prepare(self):
        self.s = session.PaintSession(bpy.context)
        session.ACTIVE = self.s
        return self.s

    def hit(self, local):
        world = self.obj.matrix_world @ Vector(local)
        position, _normal, triangle, distance = self.s.bvh.find_nearest(world)
        self.assertIsNotNone(triangle)
        self.assertLess(distance, 1e-5)
        return int(triangle), position

    def stroke(self, start=(0.12, 0.13, 0), tip=(0.7, 0.13, 0), strength=1):
        self.s.settings.strength = strength
        triangle, position = self.hit(start)
        self.s.begin_stroke(triangle, position, (20, 20))
        self.s.drag["direction"] = self.s.uv.direction_to_uv(
            triangle, np.asarray(tip, dtype=np.float64) - np.asarray(start, dtype=np.float64))
        triangle, position = self.hit(tip)
        self.s.set_stroke_tip(triangle, position)
        self.s.drag["mouse_end"] = (100, 60)
        self.s.preview()
        self.assertTrue(self.s.commit_stroke())
        return len(self.obj.data.warpflow.strokes) - 1

    def update_endpoint(self, local):
        triangle, position = self.hit(local)
        self.s.update_edit(triangle, position, (100, 100))
        self.s.preview_edit()

    def assert_endpoints(self, index, start, tip):
        actual_start, actual_tip = self.s.stroke_endpoints(index)
        np.testing.assert_allclose(actual_start, self.obj.matrix_world @ Vector(start), atol=2e-6)
        np.testing.assert_allclose(actual_tip, self.obj.matrix_world @ Vector(tip), atol=2e-6)

    def assert_matches_fresh_replay(self, expected):
        # A new session cannot retain an edit's cached prefix or draft field.
        session.end_session()
        self.prepare()
        np.testing.assert_allclose(colors(self.obj.data), expected, atol=2e-7)

    def test_new_stroke_retains_world_endpoints_after_session_reentry(self):
        self.obj.matrix_world = Matrix.Translation((2, -3, 1)) @ Matrix.Diagonal((2, 0.5, 1, 1))
        bpy.context.view_layer.update()
        self.prepare()
        self.stroke()
        self.assertTrue(self.obj.data.warpflow.strokes[0].has_tip)
        self.assert_endpoints(0, (0.12, 0.13, 0), (0.7, 0.13, 0))
        expected = colors(self.obj.data).copy()
        self.assert_matches_fresh_replay(expected)
        self.assert_endpoints(0, (0.12, 0.13, 0), (0.7, 0.13, 0))

    def test_tip_edit_previews_without_rna_changes_or_distance_solves(self):
        self.prepare()
        self.stroke()
        records = saved_records(self.obj.data)
        baseline = colors(self.obj.data).copy()
        factors = self.s.solver.stats["factorization_count"]
        solves = self.s.solver.stats["solve_count"]
        self.s.begin_edit(0, "TIP", (100, 60))
        self.update_endpoint((0.12, 0.8, 0))
        self.assertIsNotNone(self.s.edit)
        self.assertIsNone(self.s.drag)
        self.assertEqual(saved_records(self.obj.data), records)
        self.assertFalse(np.array_equal(colors(self.obj.data), baseline))
        self.assertEqual(self.s.solver.stats["solve_count"], solves)
        self.assertEqual(self.s.solver.stats["factorization_count"], factors)
        self.assertTrue(self.s.commit_edit())
        self.assertIsNone(self.s.edit)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 1)
        self.assertEqual(self.s.selected_stroke, 0)
        self.assert_endpoints(0, (0.12, 0.13, 0), (0.12, 0.8, 0))
        np.testing.assert_allclose(self.obj.data.warpflow.strokes[0].direction, (0, 1), atol=2e-6)
        self.assertEqual(self.s.solver.stats["solve_count"], solves)

    def test_handle_selection_click_within_three_pixels_does_not_snap_or_commit(self):
        self.prepare()
        self.stroke()
        self.s.area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
        before = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        ray_calls = []

        def ray(mouse):
            ray_calls.append(mouse)
            # If the controller misses its click guard, this valid ray moves
            # either endpoint to a different location and exposes the snap.
            return Vector((0.4, 0.6, 1)), Vector((0, 0, -1))

        controller = SimpleNamespace(
            paint_session=self.s,
            _mouse=lambda event: (event.mouse_x, event.mouse_y),
            _ray=ray,
        )
        for endpoint in ("START", "TIP"):
            for mouse in ((20, 20), (22, 21), (22.9, 20)):
                with self.subTest(endpoint=endpoint, mouse=mouse):
                    self.s.begin_edit(0, endpoint, (20, 20))
                    event = SimpleNamespace(mouse_x=mouse[0], mouse_y=mouse[1])
                    operators.WARPFLOW_OT_paint._update_edit(controller, event)
                    self.s.preview_edit()
                    self.assertFalse(self.s.edit["changed"])
                    self.assertFalse(self.s.edit["pending"])
                    self.assertFalse(self.s.commit_edit())
                    self.assertEqual(self.s.selected_stroke, 0)
                    self.assertEqual(saved_records(self.obj.data), records)
                    np.testing.assert_array_equal(colors(self.obj.data), before)
        self.assertEqual(ray_calls, [], "A selection click must not project a new mesh anchor")

    def test_start_edit_moves_geodesic_source_and_reuses_factorization(self):
        self.prepare()
        self.stroke()
        self.stroke((0.85, 0.85, 0), (0.2, 0.85, 0), strength=0.6)
        before_field = colors(self.obj.data).copy()
        before_source = saved_records(self.obj.data)[0][:2]
        factors = self.s.solver.stats["factorization_count"]
        solves = self.s.solver.stats["solve_count"]
        self.s.begin_edit(0, "START", (20, 20))
        triangle, position = self.hit((0.65, 0.6, 0))
        self.s.update_edit(triangle, position, (100, 100))
        self.assertEqual(self.s.solver.stats["solve_count"], solves,
                         "Mouse updates must defer distance queries until preview")
        self.s.preview_edit()
        self.assertEqual(saved_records(self.obj.data)[0][:2], before_source)
        self.assertGreater(self.s.solver.stats["solve_count"], solves)
        self.assertEqual(self.s.solver.stats["factorization_count"], factors)
        self.assertTrue(self.s.commit_edit())
        self.assertNotEqual(saved_records(self.obj.data)[0][:2], before_source)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 2)
        self.assert_endpoints(0, (0.65, 0.6, 0), (0.7, 0.13, 0))
        after_field = colors(self.obj.data).copy()
        self.assertFalse(np.allclose(after_field, before_field))
        self.assert_matches_fresh_replay(after_field)

    def test_cancel_both_handle_edits_restores_exact_field_and_metadata(self):
        self.prepare()
        self.stroke()
        self.stroke((0.8, 0.75, 0), (0.4, 0.8, 0), strength=0.35)
        before = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        for endpoint in ("START", "TIP"):
            with self.subTest(endpoint=endpoint):
                self.s.begin_edit(0, endpoint, (20, 20))
                self.update_endpoint((0.35, 0.6, 0))
                self.assertFalse(np.array_equal(colors(self.obj.data), before))
                self.s.cancel_edit()
                self.assertIsNone(self.s.edit)
                np.testing.assert_array_equal(colors(self.obj.data), before)
                self.assertEqual(saved_records(self.obj.data), records)

    def test_earlier_fractional_stroke_edit_and_delete_replay_later_strokes(self):
        self.prepare()
        self.stroke(strength=0.6)
        self.stroke((0.85, 0.2, 0), (0.85, 0.8, 0), strength=0.35)
        self.stroke((0.7, 0.85, 0), (0.2, 0.5, 0), strength=0.55)
        self.s.begin_edit(0, "TIP", (100, 60))
        self.update_endpoint((0.3, 0.7, 0))
        self.assertTrue(self.s.commit_edit())
        expected = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        self.assert_matches_fresh_replay(expected)
        self.assertEqual(saved_records(self.obj.data), records)
        self.s.selected_stroke = 1
        self.assertTrue(self.s.remove_selected())
        self.assertEqual(saved_records(self.obj.data), [records[0], records[2]])
        self.assert_matches_fresh_replay(colors(self.obj.data).copy())

    def test_delete_last_stroke_returns_all_components_to_neutral(self):
        self.prepare()
        self.stroke()
        self.s.selected_stroke = 0
        self.assertTrue(self.s.remove_selected())
        self.assertEqual(len(self.obj.data.warpflow.strokes), 0)
        np.testing.assert_array_equal(session.read_field(self.obj.data),
                                      np.zeros((len(self.obj.data.vertices), 2)))
        self.assertFalse(self.s.remove_selected())

    def test_no_selection_delete_leaves_field_and_strokes_unchanged(self):
        self.prepare()
        self.stroke()
        before = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        self.s.selected_stroke = -1
        self.assertFalse(self.s.remove_selected())
        self.assertEqual(saved_records(self.obj.data), records)
        np.testing.assert_array_equal(colors(self.obj.data), before)

    def test_proxy_endpoint_preview_then_full_commit_replays_fractional_suffix(self):
        side = 35
        points = [(x / (side - 1), y / (side - 1), 0.0)
                  for y in range(side) for x in range(side)]
        faces = []
        for y in range(side - 1):
            for x in range(side - 1):
                a = y * side + x
                faces.append((a, a + 1, a + side + 1, a + side))
        mesh = bpy.data.meshes.new("Warpflow Stroke Edit Test Dense Mesh")
        mesh.from_pydata(points, [], faces)
        mesh.update()
        uv = mesh.uv_layers.new(name="UV")
        uv.uv.foreach_set("vector", np.asarray(
            [points[loop.vertex_index][:2] for loop in mesh.loops],
            dtype=np.float32).ravel())
        self.obj.data = mesh
        bpy.context.scene.warpflow.proxy_threshold = 1000
        bpy.context.scene.warpflow.proxy_target = 200
        self.prepare()
        self.assertIsNotNone(self.s.proxy)
        self.stroke()
        self.stroke((0.85, 0.75, 0), (0.5, 0.85, 0), strength=0.4)
        factors = (self.s.solver.stats["factorization_count"],
                   self.s.proxy_solver.stats["factorization_count"])
        self.s.begin_edit(0, "START", (20, 20))
        full_solves = self.s.solver.stats["solve_count"]
        proxy_solves = self.s.proxy_solver.stats["solve_count"]
        self.update_endpoint((0.35, 0.65, 0))
        self.assertEqual(self.s.solver.stats["solve_count"], full_solves)
        self.assertGreater(self.s.proxy_solver.stats["solve_count"], proxy_solves)
        self.assertTrue(self.s.commit_edit())
        self.assertGreater(self.s.solver.stats["solve_count"], full_solves)
        self.assertEqual((self.s.solver.stats["factorization_count"],
                          self.s.proxy_solver.stats["factorization_count"]), factors)
        self.assertEqual(len(self.obj.data.warpflow.strokes), 2)
        self.assert_endpoints(0, (0.35, 0.65, 0), (0.7, 0.13, 0))
        self.assert_matches_fresh_replay(colors(self.obj.data).copy())

    def test_native_undo_redo_restores_edited_and_deleted_arrow_and_field(self):
        self.prepare()
        bpy.context.preferences.edit.use_global_undo = True
        bpy.ops.ed.undo_push(message="Warpflow Stroke Edit Test Start")
        self.stroke()
        bpy.ops.ed.undo_push(message="Warpflow Stroke Edit Test Stroke")
        original_records = saved_records(self.obj.data)
        original_field = colors(self.obj.data).copy()
        self.s.begin_edit(0, "TIP", (100, 60))
        self.update_endpoint((0.12, 0.8, 0))
        self.assertTrue(self.s.commit_edit())
        bpy.ops.ed.undo_push(message="Warpflow Stroke Edit Test Move Tip")
        edited_records = saved_records(self.obj.data)
        edited_field = colors(self.obj.data).copy()
        self.s.selected_stroke = 0
        self.assertTrue(self.s.remove_selected())
        bpy.ops.ed.undo_push(message="Warpflow Stroke Edit Test Delete")
        self.assertTrue(bpy.ops.ed.undo.poll(), "Native undo must be available for this integration test")
        self.assertIn("FINISHED", bpy.ops.ed.undo())
        self.assertIs(session.ACTIVE, self.s)
        self.assertEqual(saved_records(self.obj.data), edited_records)
        np.testing.assert_allclose(colors(self.obj.data), edited_field, atol=2e-7)
        self.assert_endpoints(0, (0.12, 0.13, 0), (0.12, 0.8, 0))
        self.assertIn("FINISHED", bpy.ops.ed.undo())
        self.assertEqual(saved_records(self.obj.data), original_records)
        np.testing.assert_allclose(colors(self.obj.data), original_field, atol=2e-7)
        self.assertIn("FINISHED", bpy.ops.ed.redo())
        self.assertEqual(saved_records(self.obj.data), edited_records)
        np.testing.assert_allclose(colors(self.obj.data), edited_field, atol=2e-7)
        self.assertIn("FINISHED", bpy.ops.ed.redo())
        self.assertEqual(len(self.obj.data.warpflow.strokes), 0)
        np.testing.assert_array_equal(session.read_field(self.obj.data),
                                      np.zeros((len(self.obj.data.vertices), 2)))

    def test_blend_persists_edited_tip_and_start_anchors(self):
        self.prepare()
        self.stroke()
        self.s.begin_edit(0, "START", (20, 20))
        self.update_endpoint((0.3, 0.6, 0))
        self.assertTrue(self.s.commit_edit())
        records = saved_records(self.obj.data)
        expected = colors(self.obj.data).copy()
        filename = str(Path(self.directory.name) / "editable-strokes.blend")
        bpy.data.libraries.write(filename, {self.obj})
        with bpy.data.libraries.load(filename, link=False) as (source, target):
            self.assertIn(self.object_name, source.objects)
            target.objects = [self.object_name]
        restored = target.objects[0]
        self.assertIsNot(restored, self.obj)
        self.assertEqual(saved_records(restored.data), records)
        np.testing.assert_array_equal(colors(restored.data), expected)
        session.end_session()
        bpy.context.scene.collection.objects.link(restored)
        self.obj.select_set(False)
        self.object_name = restored.name
        restored.select_set(True)
        bpy.context.view_layer.objects.active = restored
        self.prepare()
        self.assert_endpoints(0, (0.3, 0.6, 0), (0.7, 0.13, 0))
        np.testing.assert_allclose(colors(restored.data), expected, atol=2e-7)

    def test_legacy_stroke_has_synthetic_editable_tip_without_field_change(self):
        self.prepare()
        triangle, position = self.hit((0.2, 0.2, 0))
        self.s.begin_stroke(triangle, position, (20, 20))
        self.s.drag["direction"] = np.array((1.0, 0.0))
        self.assertTrue(self.s.commit_stroke())
        # The old format had only source, direction, and strength. Explicitly
        # clear endpoint metadata to emulate loading that saved RNA layout.
        record = self.obj.data.warpflow.strokes[0]
        record.has_tip = False
        record.tip_vertices = (0, 0, 0)
        record.tip_barycentric = (0, 0, 0)
        before = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        start, tip = self.s.stroke_endpoints(0)
        np.testing.assert_allclose(start, (0.2, 0.2, 0), atol=2e-6)
        self.assertTrue(np.all(np.isfinite(tip)))
        self.assertGreater(np.linalg.norm(np.asarray(tip) - start), 1e-6)
        self.assertEqual(saved_records(self.obj.data), records)
        np.testing.assert_array_equal(colors(self.obj.data), before)
        self.s.begin_edit(0, "TIP", (100, 60))
        self.update_endpoint((0.2, 0.8, 0))
        self.assertTrue(self.s.commit_edit())
        self.assertTrue(self.obj.data.warpflow.strokes[0].has_tip)
        self.assert_endpoints(0, (0.2, 0.2, 0), (0.2, 0.8, 0))

    def test_disconnected_component_endpoint_move_does_not_change_constraint(self):
        self.prepare()
        self.stroke()
        before = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        for endpoint in ("START", "TIP"):
            with self.subTest(endpoint=endpoint):
                self.s.begin_edit(0, endpoint, (20, 20))
                self.update_endpoint((3.2, 0.2, 0))
                self.assertFalse(self.s.commit_edit())
                self.assertEqual(saved_records(self.obj.data), records)
                np.testing.assert_array_equal(colors(self.obj.data), before)

    def test_zero_length_or_untouched_handle_drag_does_not_commit(self):
        self.prepare()
        self.stroke()
        before = colors(self.obj.data).copy()
        records = saved_records(self.obj.data)
        self.s.begin_edit(0, "TIP", (100, 60))
        self.assertFalse(self.s.commit_edit())
        self.s.begin_edit(0, "TIP", (100, 60))
        self.update_endpoint((0.12, 0.13, 0))
        self.assertFalse(self.s.commit_edit())
        self.assertEqual(saved_records(self.obj.data), records)
        np.testing.assert_array_equal(colors(self.obj.data), before)


if __name__ == "__main__":
    print("Testing editable strokes with Blender", bpy.app.version_string)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    warpflow.unregister()
    if not result.wasSuccessful():
        raise SystemExit(1)
