"""Reproducible setup/blend-only benchmarks (no geodesic solve or viewport draw).

Run: blender --background --factory-startup --python tests/benchmark_proxy_blender.py
Synthetic distance arrays isolate accumulation cost. This does not measure GPU
drawing, color-attribute uploads, heat factorization, or interactive frame rate.
"""

import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_proxy_blender import grid
from warpflow.interpolation import FieldAccumulator
from warpflow.proxy import build_proxy


def median_ms(callback, repetitions=30):
    callback()
    times = []
    for _ in range(repetitions):
        started = time.perf_counter()
        callback()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


for side, target in ((101, 2500), (201, 5000)):
    vertices, triangles = grid(side)
    components = np.zeros(len(vertices), dtype=np.int32)
    proxy = build_proxy(None, vertices, triangles, components, target_vertices=target)
    result = dict(proxy.stats)
    accumulator = FieldAccumulator(len(vertices), components, distance_scale=1.0 / side)
    distances = np.linalg.norm(vertices - vertices[0], axis=1)
    accumulator.append(distances, [1, 0])
    result["blend_1_constraint_median_ms"] = median_ms(lambda: accumulator.preview(distances, [0, 1]))
    for stroke in range(99):
        angle = stroke * 0.031
        accumulator.append(distances + stroke * 0.001, [np.cos(angle), np.sin(angle)])
    result["blend_100_constraints_median_ms"] = median_ms(lambda: accumulator.preview(distances, [0, 1]))
    proxy_field = np.tile([0.6, 0.8], (len(proxy.vertices), 1))
    result["upsample_median_ms"] = median_ms(lambda: proxy.upsample(proxy_field))
    print("WARPFLOW_PROXY_BENCHMARK " + json.dumps(result))
