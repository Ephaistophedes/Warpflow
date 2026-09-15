"""Mirror X: actual Blender surface anchors, UV frames, paired editing and undo."""
from pathlib import Path
import sys
import tempfile
import unittest

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warpflow
from warpflow import session
warpflow.register()


class MirrorTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=4)
        bpy.context.object.name = 'Mirror Test'
        bpy.context.scene.warpflow.mirror_x = True
        bpy.context.scene.warpflow.strength = 1
        bpy.context.scene.warpflow.proxy_threshold = 40000
        self.s = None

    @property
    def obj(self):
        return bpy.data.objects['Mirror Test']

    @property
    def records(self):
        return self.obj.data.warpflow.strokes

    def prepare(self):
        self.s = session.PaintSession(bpy.context)
        session.ACTIVE = self.s

    def tearDown(self):
        session.end_session()
        bpy.context.scene.warpflow.mirror_x = False

    def hit(self, point):
        world = self.obj.matrix_world @ Vector(point)
        hit, _, index, distance = self.s.bvh.find_nearest(world)
        self.assertLess(distance, 1e-5)
        return index, hit

    def draw(self, start=(.8, -.7, 0), tip=(1.4, .4, 0)):
        index, hit = self.hit(start)
        self.s.begin_stroke(index, hit, (10, 10))
        self.s.drag['direction'] = self.s.uv.direction_to_uv(index, Vector(tip) - Vector(start))
        index, hit = self.hit(tip)
        self.s.set_stroke_tip(index, hit)
        self.s.preview()
        preview = session.read_field(self.obj.data).copy()
        self.assertTrue(self.s.commit_stroke())
        np.testing.assert_allclose(session.read_field(self.obj.data), preview, atol=2e-6)

    def ends(self, index):
        return np.array([self.obj.matrix_world.inverted() @ p for p in self.s.stroke_endpoints(index)])

    def assert_pair(self, a=0, b=1):
        expected = self.ends(a) * np.array([-1, 1, 1])
        np.testing.assert_allclose(self.ends(b), expected, atol=2e-6)
        self.assertTrue(self.records[a].mirror_pair)
        self.assertEqual(self.records[a].mirror_pair, self.records[b].mirror_pair)

    def test_new_stroke_from_either_side_reflects_both_endpoints(self):
        self.prepare()
        self.draw()
        self.assert_pair()
        self.draw((-.9, .5, 0), (-1.3, 1, 0))
        self.assert_pair(2, 3)
        self.assertNotEqual(self.records[0].mirror_pair, self.records[2].mirror_pair)

    def test_mirror_off_draws_only_one_and_toggle_is_not_retroactive(self):
        self.prepare()
        self.s.settings.mirror_x = False
        self.draw()
        self.assertEqual(len(self.records), 1)
        self.s.settings.mirror_x = True
        self.assertEqual(len(self.records), 1)

    def test_local_axis_with_rotated_translated_nonuniform_object(self):
        self.obj.matrix_world = Matrix.Translation((4, -3, 2)) @ Matrix.Rotation(.7, 4, 'Z') @ Matrix.Diagonal((2, .7, 1.2, 1))
        bpy.context.view_layer.update()
        self.prepare()
        self.draw()
        self.assert_pair()

    def test_centerline_is_not_duplicated_but_crossing_stroke_is(self):
        self.prepare()
        self.draw((0, -.8, 0), (0, .8, 0))
        self.assertEqual(len(self.records), 1)
        self.assertFalse(self.records[0].mirror_pair)
        self.draw((.9, -.2, 0), (-.7, .4, 0))
        self.assertEqual(len(self.records), 3)
        self.assert_pair(1, 2)

    def test_mirrored_direction_uses_other_uv_frame(self):
        # Each triangle can own a different UV frame. Rotate only negative-X
        # triangles to ensure mirroring a raw RG component would be incorrect.
        mesh = self.obj.data
        for poly in mesh.polygons:
            if poly.center.x < 0:
                for li in poly.loop_indices:
                    old = mesh.uv_layers.active.uv[li].vector.copy()
                    mesh.uv_layers.active.uv[li].vector = (old.y, 1 - old.x)
        self.prepare()
        self.draw()
        mirrored = self.records[1]
        triangle = self.s._source_triangle(mirrored.vertices)
        start, tip = self.s.stroke_endpoints(1)
        expected = self.s.uv.direction_to_uv(triangle, self.s.inverse_matrix.to_3x3() @ (tip - start))
        np.testing.assert_allclose(mirrored.direction, expected, atol=1e-6)
        self.assertFalse(np.allclose(mirrored.direction, np.asarray(self.records[0].direction) * [-1, 1]))

    def test_edit_pair_onto_centerline_removes_duplicate(self):
        self.prepare()
        self.draw((0, -.8, 0), (.7, .4, 0))
        self.assertEqual(len(self.records), 2)
        self.s.begin_edit(0, 'TIP', (10, 10))
        tri, hit = self.hit((0, .9, 0))
        self.s.update_edit(tri, hit, (50, 50))
        self.s.preview_edit()
        preview = session.read_field(self.obj.data).copy()
        self.s.commit_edit()
        self.assertEqual(len(self.records), 1)
        self.assertFalse(self.records[0].mirror_pair)
        np.testing.assert_allclose(session.read_field(self.obj.data), preview, atol=2e-6)

    def test_proxy_mirrored_preview_and_full_commit(self):
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=40, y_subdivisions=40, size=4)
        bpy.context.object.name = 'Mirror Test'
        bpy.context.scene.warpflow.proxy_threshold = 1000
        bpy.context.scene.warpflow.proxy_target = 300
        self.prepare()
        self.assertIsNotNone(self.s.proxy)
        tri, start = self.hit((.8, -.7, 0))
        self.s.begin_stroke(tri, start, (10, 10))
        self.s.drag['direction'] = self.s.uv.direction_to_uv(tri, Vector((.5, 1.1, 0)))
        tri, tip = self.hit((1.3, .4, 0))
        self.s.set_stroke_tip(tri, tip)
        self.s.preview()
        self.assertEqual(len(self.records), 0)
        self.s.commit_stroke()
        self.assert_pair()
        self.assertEqual(self.s.solver.stats['factorization_count'], 2)
        self.assertEqual(self.s.proxy_solver.stats['factorization_count'], 2)
        committed = session.read_field(self.obj.data).copy()
        self.s.rebuild()
        # Saved RNA vectors are float32, while the preview/solver uses float64.
        np.testing.assert_allclose(session.read_field(self.obj.data), committed, atol=2e-6)

    def test_either_member_tip_and_start_edit_update_pair_without_refactorization(self):
        self.prepare()
        self.draw()
        factors = self.s.solver.stats['factorization_count']
        for index, endpoint, target in ((1, 'TIP', (-1.1, .9, 0)), (0, 'START', (.5, -.9, 0))):
            self.assertTrue(self.s.begin_edit(index, endpoint, (10, 10)))
            tri, hit = self.hit(target)
            self.assertTrue(self.s.update_edit(tri, hit, (50, 50)))
            self.s.preview_edit()
            preview = session.read_field(self.obj.data).copy()
            self.assertTrue(self.s.commit_edit())
            self.assertEqual(len(self.records), 2)
            self.assert_pair()
            np.testing.assert_allclose(session.read_field(self.obj.data), preview, atol=2e-6)
        self.assertEqual(self.s.solver.stats['factorization_count'], factors)

    def test_partial_strength_mirror_edit_replays_later_history(self):
        self.prepare()
        self.s.settings.strength = .4
        self.draw()
        self.draw((.5, .4, 0), (1.6, .6, 0))
        self.s.begin_edit(1, 'TIP', (10, 10))
        tri, hit = self.hit((-1.2, 1.2, 0))
        self.s.update_edit(tri, hit, (50, 50))
        self.s.preview_edit()
        preview = session.read_field(self.obj.data).copy()
        self.s.commit_edit()
        np.testing.assert_allclose(session.read_field(self.obj.data), preview, atol=2e-6)
        self.assert_pair()

    def test_delete_pair_and_native_undo_redo_restores_links(self):
        self.prepare()
        bpy.context.preferences.edit.use_global_undo = True
        bpy.ops.ed.undo_push(message='Mirror baseline')
        self.draw()
        bpy.ops.ed.undo_push(message='Mirrored stroke')
        self.s.selected_stroke = 1
        self.assertTrue(self.s.remove_selected())
        bpy.ops.ed.undo_push(message='Delete mirrored pair')
        self.assertEqual(len(self.records), 0)
        np.testing.assert_array_equal(session.read_field(self.obj.data), 0)
        bpy.ops.ed.undo()
        self.assertEqual(len(self.records), 2)
        self.assert_pair()
        bpy.ops.ed.undo()
        self.assertEqual(len(self.records), 0)
        bpy.ops.ed.redo()
        self.assertEqual(len(self.records), 2)
        self.assert_pair()

    def test_off_edit_unlinks_without_changing_other_arrow(self):
        self.prepare()
        self.draw()
        other = self.ends(1).copy()
        self.s.settings.mirror_x = False
        self.s.begin_edit(0, 'TIP', (10, 10))
        tri, hit = self.hit((.4, 1.1, 0))
        self.s.update_edit(tri, hit, (50, 50))
        self.s.commit_edit()
        np.testing.assert_allclose(self.ends(1), other)
        self.assertFalse(self.records[0].mirror_pair)
        self.assertFalse(self.records[1].mirror_pair)
        self.s.selected_stroke = 0
        self.s.remove_selected()
        self.assertEqual(len(self.records), 1)

    def test_mirror_edit_existing_unpaired_stroke_creates_partner(self):
        self.prepare()
        self.s.settings.mirror_x = False
        self.draw()
        self.s.settings.mirror_x = True
        self.s.begin_edit(0, 'TIP', (10, 10))
        tri, hit = self.hit((1.2, .9, 0))
        self.s.update_edit(tri, hit, (50, 50))
        self.s.preview_edit()
        self.s.commit_edit()
        self.assertEqual(len(self.records), 2)
        self.assert_pair()

    def test_cancel_and_save_restore_pair_metadata(self):
        self.prepare()
        self.draw()
        before = session.read_field(self.obj.data).copy()
        self.s.begin_edit(0, 'TIP', (10, 10))
        tri, hit = self.hit((1.2, 1.2, 0))
        self.s.update_edit(tri, hit, (50, 50))
        self.s.preview_edit()
        self.s.cancel_edit()
        self.assert_pair()
        np.testing.assert_array_equal(session.read_field(self.obj.data), before)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'mirror.blend')
            bpy.data.libraries.write(path, {self.obj})
            with bpy.data.libraries.load(path) as (_, target):
                target.objects = [self.obj.name]
            loaded = target.objects[0]
            self.assertEqual(loaded.data.warpflow.strokes[0].mirror_pair, self.records[0].mirror_pair)
            self.assertEqual(loaded.data.warpflow.strokes[1].mirror_pair, self.records[0].mirror_pair)

    def test_missing_opposite_surface_keeps_original_with_notice(self):
        for vertex in self.obj.data.vertices:
            vertex.co.x += 5
        self.obj.data.update()
        self.prepare()
        self.draw((4.5, -.7, 0), (5.5, .4, 0))
        self.assertEqual(len(self.records), 1)
        self.assertIn('no matching opposite surface', self.s.notice)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MirrorTests))
warpflow.unregister()
if not result.wasSuccessful():
    raise SystemExit(1)
