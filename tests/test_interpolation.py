"""Numerical tests run outside Blender: python -m unittest discover -s tests."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


# Import the numerical module without importing Blender registration code.
_path = Path(__file__).resolve().parents[1] / "warpflow" / "interpolation.py"
_spec = importlib.util.spec_from_file_location("warpflow_interpolation_test", _path)
interpolation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(interpolation)
FieldAccumulator = interpolation.FieldAccumulator


class InterpolationTests(unittest.TestCase):
    def test_unpainted_encodes_neutral(self):
        field = FieldAccumulator(8)
        np.testing.assert_array_equal(field.field, np.zeros((8, 2)))
        colors = interpolation.encode_colors(field.field)
        np.testing.assert_array_equal(colors[:, :2], np.full((8, 2), 0.5))
        np.testing.assert_array_equal(colors[:, 2], 0.0)
        np.testing.assert_array_equal(colors[:, 3], 1.0)
        np.testing.assert_array_equal(interpolation.decode_colors(colors), field.field)

    def test_one_stroke_covers_disconnected_components(self):
        field = FieldAccumulator(6, [0, 0, 0, 1, 1, 2])
        result = field.append([0.0, 1.0, 1.0e20, np.inf, np.inf, np.inf], [3, 4])
        np.testing.assert_allclose(result, np.tile([0.6, 0.8], (6, 1)))
        np.testing.assert_array_equal(field.weight_sum[3:], 0.0)

    def test_all_constraints_match_direct_weighted_sum(self):
        rng = np.random.default_rng(73)
        n_vertices = 250
        accumulator = FieldAccumulator(n_vertices, sharpness=1.7, distance_scale=0.4)
        direct_sum = np.zeros((n_vertices, 2))
        for _ in range(40):
            distances = rng.uniform(0, 100, n_vertices)
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            weights = (0.4 / (distances + 0.4)) ** 1.7
            direct_sum += weights[:, None] * direction
            accumulator.append(distances, direction)
        expected = direct_sum / np.linalg.norm(direct_sum, axis=1)[:, None]
        np.testing.assert_allclose(accumulator.field, expected, atol=1.0e-11)

    def test_preview_is_pure_and_matches_commit(self):
        field = FieldAccumulator(4)
        field.append([0, 1, 2, 3], [1, 0])
        before = field.field.copy()
        weights = field.weight_sum.copy()
        preview = field.preview([3, 2, 1, 0], [0, 1], 0.35)
        np.testing.assert_array_equal(field.field, before)
        np.testing.assert_array_equal(field.weight_sum, weights)
        self.assertEqual(field.count, 1)
        committed = field.append([3, 2, 1, 0], [0, 1], 0.35)
        np.testing.assert_array_equal(preview, committed)
        np.testing.assert_allclose(np.linalg.norm(committed, axis=1), 1.0)

    def test_strength_blends_against_pre_stroke_field(self):
        field = FieldAccumulator(1)
        field.append([0], [1, 0])
        raw = np.asarray([1, 1]) / np.sqrt(2)
        expected = 0.75 * np.asarray([1, 0]) + 0.25 * raw
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(field.append([0], [0, 1], 0.25)[0], expected)
        # A same-direction following stroke retains the materialized strength.
        field.append([0], expected)
        np.testing.assert_allclose(field.field[0], expected)

    def test_zero_strength_is_noop_including_later_strokes(self):
        field = FieldAccumulator(2)
        field.append([0, 2], [0, 1], 0)
        np.testing.assert_array_equal(field.field, 0.0)
        np.testing.assert_array_equal(field.weight_sum, 0.0)
        field.append([2, 0], [1, 0])
        np.testing.assert_array_equal(field.field, [[1, 0], [1, 0]])

    def test_components_stop_using_remote_seed_when_locally_painted(self):
        field = FieldAccumulator(6, [0, 0, 1, 1, 2, 2])
        field.append([0, 1, np.inf, np.inf, np.inf, np.inf], [1, 0])
        field.append([np.inf, np.inf, 0, 1, np.inf, np.inf], [0, 1])
        np.testing.assert_array_equal(field.field, [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])
        field.append([1, 0, np.inf, np.inf, np.inf, np.inf], [-1, 0])
        np.testing.assert_array_equal(field.field[2:], [[0, 1], [0, 1], [1, 0], [1, 0]])

    def test_opposite_equal_constraints_do_not_create_unpainted_gap(self):
        field = FieldAccumulator(3)
        field.append([0, 1, 100], [1, 0])
        field.append([0, 1, 100], [-1, 0])
        np.testing.assert_array_equal(field.field, [[1, 0], [1, 0], [1, 0]])

    def test_tiny_distant_weights_still_normalize(self):
        field = FieldAccumulator(2, sharpness=8)
        field.append([0, 1.0e40], [0, -1])
        np.testing.assert_array_equal(field.field, [[0, -1], [0, -1]])
        field.append([0, 1.0e40], [1, 0])
        np.testing.assert_allclose(field.field, np.tile(np.asarray([1, -1]) / np.sqrt(2), (2, 1)))

    def test_replaying_history_reproduces_strength_and_undo(self):
        strokes = [([0, 1, 2], [1, 0], 1.0), ([1, 1, 0], [0, 1], 0.2), ([2, 0, 1], [-1, 1], 0.7)]
        field = FieldAccumulator(3)
        states = [field.field.copy()]
        for stroke in strokes:
            states.append(field.append(*stroke).copy())
        undo = FieldAccumulator(3)
        for stroke in strokes[:-1]:
            undo.append(*stroke)
        np.testing.assert_array_equal(undo.field, states[-2])
        undo.append(*strokes[-1])
        np.testing.assert_array_equal(undo.field, states[-1])

    def test_invalid_input_fails_without_mutation(self):
        field = FieldAccumulator(2)
        with self.assertRaises(ValueError):
            field.append([0, 1], [0, 0])
        with self.assertRaises(ValueError):
            field.append([np.inf, np.inf], [1, 0])
        with self.assertRaises(ValueError):
            field.append([-1, 1], [1, 0])
        np.testing.assert_array_equal(field.field, 0.0)


if __name__ == "__main__":
    unittest.main()
