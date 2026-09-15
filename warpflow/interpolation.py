"""Incremental inverse geodesic distance interpolation.

All constraints participate: keeping the weighted vector sum gives O(V) work and
O(V) memory for a new stroke, independent of the number of committed constraints.
An O(V*C) distance/constraint matrix and approximate KD-tree truncation are both
unnecessary. The session separately uses a BVH to locate stroke sources.

Strength is applied against the *pre-stroke displayed field*. A fractional
commit materializes that blend in the cached sum, retaining its length. This
preserves earlier strength decisions when more strokes are added. With all
strengths at one this is the ordinary exact inverse-distance weighted average.
Replaying completed strokes in order reproduces the same result after undo.
"""

from __future__ import annotations

import numpy as np


_EPS = 1.0e-12


def normalize_vectors(vectors, fallback=None):
    """Normalize (..., 2) vectors; neutral zeros stay zero unless a fallback exists."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[-1:] != (2,):
        raise ValueError("Flow vectors must have exactly two components")
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    result = np.divide(values, lengths, out=np.zeros_like(values), where=lengths > _EPS)
    if fallback is not None:
        alternate = np.broadcast_to(np.asarray(fallback, dtype=np.float64), values.shape)
        alternate_lengths = np.linalg.norm(alternate, axis=-1, keepdims=True)
        alternate = np.divide(
            alternate, alternate_lengths, out=np.zeros_like(alternate),
            where=alternate_lengths > _EPS,
        )
        result = np.where(lengths > _EPS, result, alternate)
    return result


def encode_colors(field):
    """RGBA float colors: direction in RG, B=0, A=1; zero field is neutral."""
    direction = normalize_vectors(field)
    colors = np.empty((len(direction), 4), dtype=np.float32)
    colors[:, :2] = direction * 0.5 + 0.5
    colors[:, 2] = 0.0
    colors[:, 3] = 1.0
    return colors


def decode_colors(colors):
    """Decode the per-vertex RG source of truth, retaining unpainted neutral zeros."""
    values = np.asarray(colors, dtype=np.float64)
    return normalize_vectors(values[:, :2] * 2.0 - 1.0)


class FieldAccumulator:
    """Cache committed field state and preview exactly one temporary constraint.

    ``distances`` is length V, with infinity outside the source component.
    A disconnected component with no local constraint inherits the first stroke
    direction as a deterministic coverage seed. It never acquires finite-distance
    influence from a different component. Once painted locally it uses only its
    own constraints; its first partial-strength stroke blends against that seed.

    ``distance_scale`` is normally the mesh's mean edge length. The bounded
    inverse-distance kernel ``(scale / (distance + scale)) ** sharpness`` avoids
    singular weights at sources while preserving smooth full-surface support.
    """

    def __init__(self, n_vertices, components=None, sharpness=2.0, distance_scale=1.0):
        self.n_vertices = int(n_vertices)
        if self.n_vertices < 0:
            raise ValueError("Vertex count must be nonnegative")
        if components is None:
            components = np.zeros(self.n_vertices, dtype=np.int32)
        self.components = np.asarray(components, dtype=np.int32).copy()
        if self.components.shape != (self.n_vertices,):
            raise ValueError("Component labels must have one entry per vertex")
        self.sharpness = float(sharpness)
        self.distance_scale = float(distance_scale)
        if not np.isfinite(self.sharpness) or self.sharpness <= 0:
            raise ValueError("Influence sharpness must be positive and finite")
        if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("Distance scale must be positive and finite")
        self.clear()

    def clear(self):
        self._vector_sum = np.zeros((self.n_vertices, 2), dtype=np.float64)
        self._weight_sum = np.zeros(self.n_vertices, dtype=np.float64)
        self._field = np.zeros((self.n_vertices, 2), dtype=np.float64)
        self._seed = None
        self.count = 0

    @property
    def field(self):
        """Current normalized field. Callers must not modify the returned array."""
        return self._field

    @property
    def weight_sum(self):
        return self._weight_sum

    def _calculate(self, distances, direction2, strength):
        distances = np.asarray(distances, dtype=np.float64)
        if distances.shape != (self.n_vertices,):
            raise ValueError("Distances must have one entry per vertex")
        direction = np.asarray(direction2, dtype=np.float64)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("Stroke direction must be a finite 2D vector")
        length = np.linalg.norm(direction)
        if length <= _EPS:
            raise ValueError("Stroke direction must be nonzero")
        direction = direction / length
        strength = float(strength)
        if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("Stroke strength must be between zero and one")
        if strength == 0.0:
            return self._field.copy(), self._vector_sum.copy(), self._weight_sum.copy(), self._seed
        finite = np.isfinite(distances)
        if np.any(distances[finite] < -1.0e-8):
            raise ValueError("Geodesic distances must be nonnegative")
        if not np.any(finite):
            raise ValueError("Stroke has no reachable vertices")
        weights = np.zeros(self.n_vertices, dtype=np.float64)
        # Log evaluation is stable for long meshes and large falloff exponents.
        log_weights = -self.sharpness * np.log1p(np.maximum(distances[finite], 0) / self.distance_scale)
        weights[finite] = np.exp(np.maximum(log_weights, -600.0))
        summed = self._vector_sum + weights[:, None] * direction
        # hypot rescales internally: sum-of-squares norms underflow for the
        # extremely small weights retained at distant vertices.
        magnitudes = np.hypot(summed[:, 0], summed[:, 1])
        previous = self._field
        fallback = np.where(np.linalg.norm(previous, axis=1)[:, None] > _EPS, previous, direction)
        # Relative tolerance: tiny but coherent weights on distant vertices must
        # still normalize. Only near-cancellation relative to total influence
        # requires deterministic old-direction fallback.
        totals = self._weight_sum + weights
        valid = magnitudes > np.maximum(totals * _EPS, np.finfo(np.float64).tiny)
        candidate = np.divide(summed, magnitudes[:, None], out=fallback.copy(), where=valid[:, None])
        blended = normalize_vectors((1.0 - strength) * previous + strength * candidate, fallback)
        field = previous.copy()
        field[finite] = blended[finite]
        seed = self._seed if self._seed is not None else direction.copy()
        unconstrained = totals == 0.0
        field[unconstrained] = seed
        if strength < 1.0:
            # Store the committed orientation so a later stroke retains this
            # strength operation; retaining magnitude keeps strength=1 exact.
            summed[finite] = field[finite] * magnitudes[finite, None]
        return field, summed, totals, seed

    def preview(self, distances, direction2, strength=1.0):
        """O(V) preview without mutating any committed state."""
        return self._calculate(distances, direction2, strength)[0]

    def append(self, distances, direction2, strength=1.0):
        """Commit one single-point stroke and return its displayed field.

        V1 uses one surface point per stroke. A path-sampled stroke could add
        several source terms before making one strength blend/undo boundary.
        """
        field, summed, totals, seed = self._calculate(distances, direction2, strength)
        self._field = field
        self._vector_sum = summed
        self._weight_sum = totals
        self._seed = seed
        self.count += 1
        return self._field
