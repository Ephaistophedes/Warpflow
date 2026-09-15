"""Reproduce numerical Warpflow timings without registering the add-on.

System Python:
    python scripts/benchmark.py --output tests/benchmark_results.json
Blender:
    blender --background --factory-startup --python scripts/benchmark.py -- \
        --output tests/benchmark_results.json --require-vendored-scipy

Regular grids are useful reproducible workloads, not a promise of interactive
viewport performance. These measurements exclude Blender attribute writes, GPU
drawing, ray casts, UV setup, decimated proxy construction and texture baking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[1]


def load_numerical_modules():
    # Do not import warpflow.__init__: packaging and registration are independent
    # from this numerical benchmark. Relative imports still work in this alias.
    package_name = "_warpflow_numerical_benchmark"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "warpflow")]
    sys.modules[package_name] = package
    importlib.import_module(package_name + ".dependencies").prepare()
    geodesic = importlib.import_module(package_name + ".geodesic")
    interpolation = importlib.import_module(package_name + ".interpolation")
    return geodesic, interpolation


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def summarize(milliseconds, np):
    return {
        "median_ms": round(float(np.median(milliseconds)), 4),
        "min_ms": round(min(milliseconds), 4),
        "max_ms": round(max(milliseconds), 4),
        "samples_ms": [round(value, 4) for value in milliseconds],
    }


def run_grid(side, queries, geodesic, interpolation, np):
    axis = np.linspace(-1.0, 1.0, side)
    x, y = np.meshgrid(axis, axis)
    vertices = np.stack((x.ravel(), y.ravel(), np.zeros(side * side)), axis=1)
    corners = np.arange(side * side - side, dtype=np.int64)
    corners = corners[corners % side < side - 1]
    triangles = np.concatenate((
        np.stack((corners, corners + 1, corners + side + 1), axis=1),
        np.stack((corners, corners + side + 1, corners + side), axis=1),
    ))
    started = time.perf_counter()
    solver = geodesic.create_solver(vertices, triangles)
    setup_ms = (time.perf_counter() - started) * 1000.0
    if not solver.backend.startswith("Heat method"):
        raise RuntimeError("Benchmark requires the heat backend: " + solver.backend)
    started = time.perf_counter()
    accumulator = interpolation.FieldAccumulator(
        len(vertices), solver.components, sharpness=2.0,
        distance_scale=solver.mean_edge_length,
    )
    accumulator_setup_ms = (time.perf_counter() - started) * 1000.0
    # Two committed constraints make the preview exercise the same incremental
    # accumulator used by an active session. Setup/replay is outside query timing.
    accumulator.append(solver.distances([side * side // 2]), [1.0, 0.0])
    accumulator.append(solver.distances([side * (side // 3) + side // 3]), [0.0, 1.0])
    rng = np.random.default_rng(42)
    face_indices = rng.integers(0, len(triangles), size=queries + 1)
    barycentric_weights = np.array([0.2, 0.3, 0.5])
    warm = solver.distances(triangles[face_indices[0]], barycentric_weights)
    accumulator.preview(warm, [-0.6, 0.8], strength=0.65)
    factorizations = solver.stats["factorization_count"]
    adjacency_builds = solver.stats["adjacency_builds"]
    solve_samples, blend_samples, encode_samples = [], [], []
    for index, face_index in enumerate(face_indices[1:]):
        started = time.perf_counter()
        distances = solver.distances(triangles[face_index], barycentric_weights)
        solve_samples.append((time.perf_counter() - started) * 1000.0)
        angle = 0.7 + index * 0.2
        direction = np.array([np.cos(angle), np.sin(angle)])
        started = time.perf_counter()
        field = accumulator.preview(distances, direction, strength=0.65)
        blend_samples.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        colors = interpolation.encode_colors(field)
        encode_samples.append((time.perf_counter() - started) * 1000.0)
        if not (np.all(np.isfinite(distances)) and np.all(np.isfinite(colors))
                and np.allclose(np.linalg.norm(field, axis=1), 1.0, atol=1e-8)):
            raise AssertionError("Nonfinite or unnormalized benchmark result")
    if solver.stats["factorization_count"] != factorizations:
        raise AssertionError("Query rebuilt a sparse factorization")
    if solver.stats["adjacency_builds"] != adjacency_builds:
        raise AssertionError("Query rebuilt mesh adjacency")
    if solver.stats["graph_solve_count"]:
        raise AssertionError("Heat workload unexpectedly used a graph fallback")
    return {
        "grid_side": side,
        "vertices": len(vertices),
        "triangles": len(triangles),
        "backend": solver.backend,
        "solver_setup_ms": round(setup_ms, 4),
        "accumulator_setup_ms": round(accumulator_setup_ms, 4),
        "solve": summarize(solve_samples, np),
        "preview_blend": summarize(blend_samples, np),
        "rgba_encode": summarize(encode_samples, np),
        "solve_blend_encode_median_sum_ms": round(
            float(np.median(solve_samples) + np.median(blend_samples)
                  + np.median(encode_samples)), 4),
        "factorization_count": solver.stats["factorization_count"],
        "factor_nonzeros": solver.stats.get("factor_nnz"),
        "adjacency_builds": solver.stats["adjacency_builds"],
        "heat_solve_count": solver.stats["heat_solve_count"],
        "graph_solve_count": solver.stats["graph_solve_count"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write this JSON file")
    parser.add_argument("--sides", type=int, nargs="+", default=[51, 101, 201, 317])
    parser.add_argument("--queries", type=int, default=7)
    parser.add_argument("--require-vendored-scipy", action="store_true")
    # Blender consumes the arguments preceding --; ordinary Python does not.
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.queries < 1 or any(side < 2 for side in args.sides):
        parser.error("queries must be positive and grid sides must be at least 2")
    started = time.perf_counter()
    geodesic, interpolation = load_numerical_modules()
    import numpy as np
    import scipy
    import scipy.sparse.linalg  # Exclude its lazy startup cost from mesh scaling.
    dependency_prepare_ms = (time.perf_counter() - started) * 1000.0
    scipy_path = Path(scipy.__file__).resolve()
    vendor_root = (ROOT / "warpflow" / "_vendor").resolve()
    vendored = scipy_path.is_relative_to(vendor_root)
    if args.require_vendored_scipy and not vendored:
        raise RuntimeError(f"Expected bundled SciPy; loaded {scipy_path}")
    try:
        import bpy
        blender_version = bpy.app.version_string
    except ImportError:
        blender_version = None
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "cpu": cpu_name(),
            "python": platform.python_version(),
            "blender": blender_version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scipy_path": str(scipy_path),
            "scipy_vendored": vendored,
        },
        "methodology": {
            "mesh": "Regular triangulated planar grid spanning [-1,1] squared",
            "queries_per_mesh": args.queries,
            "random_seed": 42,
            "source": "Face point with barycentric weights [0.2, 0.3, 0.5]",
            "committed_constraints_before_preview": 2,
            "preview_strength": 0.65,
            "influence_sharpness": 2.0,
            "warmup": "NumPy and scipy.sparse.linalg imported before setup; one unmeasured preview per mesh",
            "excludes": [
                "Blender color-attribute writes", "GPU drawing", "ray casts",
                "UV setup", "decimated proxy construction", "texture baking",
            ],
            "scope": "Numerical benchmarks only; no full viewport interactivity claim",
        },
        "dependency_prepare_ms": round(dependency_prepare_ms, 4),
        "meshes": [],
    }
    for side in args.sides:
        measurement = run_grid(side, args.queries, geodesic, interpolation, np)
        result["meshes"].append(measurement)
        print(f"Measured {measurement['vertices']:,} vertices", file=sys.stderr, flush=True)
    encoded = json.dumps(result, indent=2)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
