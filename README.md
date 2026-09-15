# Warpflow 1.0

Paint a few directional strokes on a mesh to create a complete flow field, then bake its vertex colors to a PNG flowmap. Warpflow targets **Blender 5.0+** and uses `mesh.color_attributes`, current UV attributes, and the `gpu` drawing API.

Runtime validation was performed in **Blender 5.2.1 LTS on Windows x64**. Blender 5.0 is the API compatibility target; it has not been separately runtime-tested.

## Install

1. In Blender, open **Edit → Preferences → Add-ons**.
2. Open the menu at the top right, choose **Install from Disk**, and select [`dist/Warpflow-1.0.0-windows-x64.zip`](dist/Warpflow-1.0.0-windows-x64.zip).
3. Enable **Warpflow**. In the 3D View, press **N** and open the **Warpflow** tab.

This is a legacy add-on ZIP, with a `warpflow/` package at its root. Select the ZIP without extracting it. See Blender's [legacy add-on installation instructions](https://docs.blender.org/manual/en/5.0/editors/preferences/addons.html#installing-legacy-add-ons).

The Windows package includes private **SciPy 1.16.2** binaries for CPython **3.11 (`cp311`)** and **3.13 (`cp313`)**, using Blender's NumPy. No `pip`, download, or Blender Python modification occurs at installation or runtime. An already available host SciPy takes priority. The two Python versions accommodate the [Python 3.13 upgrade in Blender 5.1](https://developer.blender.org/docs/release_notes/5.1/python_api/#python-3-13).

This is the initial Warpflow release. Restart Blender before updating the installer in the future, so Windows releases the bundled numerical-library DLLs for replacement.

The bundled binaries are Windows x64 only. Other platforms or Python ABIs can use an existing compatible SciPy; otherwise the explicitly labelled graph approximation described below is used. Those platforms have not been tested.

## Paint

Prepare one local, editable mesh in **Object Mode**, with an existing active UV unwrap. Disable its viewport modifiers before painting. Make shared mesh data single-user through **Object → Relations → Make Single User**. Faces must have nonzero area and usable, noncollapsed UV triangles.

1. Click **Enter Flow Paint Mode**. The progress cursor and status text cover the one-time surface, UV, and preview preparation.
2. **Left-click and drag on the surface**, then release. A drag of at least a few pixels creates one directional constraint at its **start point**.
3. The first stroke fills the entire mesh with a uniform UV-space direction. Add strokes in other places and directions to shape the interpolated field.
4. Adjust **Influence Sharpness** for tighter or broader transitions. **Strength** applies to the next stroke. Toggle **In-progress Stroke**, **Saved Stroke Arrows**, **Direction Arrows**, and **Flowmap Colors** as needed.
5. Press **Esc** or click **Exit Flow Paint Mode**, then export.

| Control | Result |
| --- | --- |
| Left-drag / release | Preview / commit one stroke |
| Click a saved arrow | Select that stroke; the selected arrow turns orange |
| Drag round start handle | Move the constraint source, keeping its tip fixed |
| Drag square tip handle | Change the direction, keeping its source fixed |
| Delete / X | Delete the selected stroke and recompute the field |
| Ctrl-drag | Create a new stroke even over an existing arrow |
| Right-click | Cancel the unfinished stroke or endpoint edit and keep painting |
| Esc during an endpoint edit | Cancel that edit and keep painting |
| Esc otherwise | Cancel the unfinished new stroke and exit; keep committed strokes |
| Ctrl-Z / Ctrl-Shift-Z | Native undo / redo, one step per committed stroke, endpoint edit, or deletion |
| Normal orbit, pan, zoom | Navigate during painting |
| Clear Strokes | Reset to neutral outside paint mode; undoable |

Changing Influence Sharpness rebuilds the stored constraints and creates its own undo step. Entering mode establishes the undo baseline. Preview updates and cancelling a drag do not create stroke undo steps.

Geometry, object transforms, and UVs are cached for the session. If they change after painting, use **Clear Strokes** before starting a new field. Save the `.blend` to retain both colors and stroke metadata.

## Edit saved strokes

Completed arrows remain visible on the active mesh after leaving paint mode. Re-enter **Flow Paint Mode** to select and edit them. Click the arrow shaft to select it, or drag either handle directly. The round handle is the stroke's start; the square handle is its tip. Use **Delete Selected Stroke** in the sidebar or press **Delete/X**. Deleting the last stroke restores the neutral field. Selection itself creates no undo step.

Both endpoints are saved as triangle/barycentric anchors. Moving the start updates surface influence and direction; moving the tip updates direction. Arrow length remains a visual handle and does not encode strength or magnitude. The field updates during editing and the original strength and history order are preserved. Release commits the full-resolution result; cancellation restores the exact previous colors.

New strokes and endpoint edits retain the last valid position when the pointer leaves the original connected surface, so releasing off-mesh does not replace the arrow with a floating endpoint. Guides connect the two anchors with a straight visual arrow; they are not sampled paths draped over curved surfaces. Hidden surface endpoints cannot be selected through the mesh. **Saved Stroke Arrows** toggles their display; outside paint mode they also follow Blender's viewport overlay visibility.

## Mirror X

Turn **Mirror X** on below Strength to mirror each new stroke automatically across the object's **local X=0 plane**. Paint on either side; there is no positive/negative side setting. The object origin defines the plane, including when the object is translated, rotated or scaled. Both the start and tip are reflected, and the mirrored direction is encoded using the destination surface's UV tangent frame.

The live preview includes the mirrored field and arrow. Release creates a linked pair in one undo step. With Mirror X on, selecting either arrow highlights both; moving either endpoint updates the other arrow, and Delete removes the pair. An arrow lying entirely on the center plane is stored once. Editing an older unpaired stroke while Mirror X is on adds its counterpart on commit.

Turning Mirror X off leaves existing strokes in place and creates subsequent strokes individually. Editing a pair member with Mirror X off changes only that arrow and unlinks the pair; deletion removes only the selected member. Toggling alone does not retroactively duplicate existing strokes. The setting at mouse-down controls the current paint/edit gesture.

Mirroring uses the base mesh and snaps to the nearest reflected surface within 2% of the local mesh diagonal. A matching surface must exist on the opposite side; missing counterparts are not projected back onto the original side. A new stroke without a match is kept individually with a status notice. A paired endpoint edit without a valid match retains its previous position. The two constraints use the existing ordered-strength model (drawn stroke followed by its copy); partial-strength results retain that model's order dependence. Source solves reuse the session's factors, including proxy previews.

## Export a flowmap

Exit paint mode, select the painted mesh, and open **Export Flowmap** in the same sidebar. Choose **512**, **1024**, **2048**, or **Custom** resolution; custom UI values range from 64 to 8192. Set the padding in pixels, click **Export Flowmap**, and choose a PNG filepath.

Export uses the **active UV map**, packed within the single `[0, 1]` tile. UDIM and out-of-tile UVs are rejected. Overlapping UVs follow Blender's native bake behavior, so unwrap without unintended overlap. The export is the **painted base mesh**, without evaluated modifiers.

Blender's **Cycles emission bake** rasterizes the `Warpflow` color attribute, with native UV interpolation and padding. A temporary scene, object, mesh, material, and image isolate the bake from the artist's render settings and materials. Cycles startup is slower than a custom rasterizer, but this reuses Blender's established bake pipeline. Output replaces the selected file only after a successful save.

The result is a **16-bit RGBA PNG containing data values**. Import it into a shader or engine as **Non-Color / linear data**, without an sRGB decode. Decode direction as `RG * 2 - 1`; normalize nonneutral samples when using them. Uncovered pixels are neutral except for bake padding. Export normalizes interpolated pixel directions, while exact or near cancellation inside a triangle remains neutral because it has no defined direction. PNG quantization introduces a small error, including at neutral values; use a small zero tolerance when decoding.

## Field representation and behavior

The field is stored in the mesh's **`Warpflow` color attribute**, with domain **`POINT`** and type **`FLOAT_COLOR`**:

```text
R = direction_u * 0.5 + 0.5
G = direction_v * 0.5 + 0.5
B = 0.0
A = 1.0
```

Painted vertex directions have unit length. Magnitude is not part of the model. Zero strokes produce the neutral value **`(0.5, 0.5, 0.0, 1.0)`**. At an exact cancellation between opposing constraints, the previous vertex direction is retained deterministically, preventing an unpainted vertex gap.

Rendering and export read the color attribute. Saved stroke metadata allows reconstruction after undo, re-entry, or a sharpness change. Re-entering paint mode regenerates colors from that history; direct manual edits to the attribute are therefore suitable for export but are not retained by a subsequent reconstruction.

**Strength** blends the pre-stroke displayed field toward the new interpolated result, then normalizes. A zero-strength stroke has no effect. The first nonzero-strength stroke still establishes a unit direction everywhere. Fractional strength is materialized into the cached weighted field, preserving that decision when later strokes arrive. Consequently, partial-strength histories are intentionally order-dependent; they are not equivalent to a fresh unmodified weighted sum of the original directions. With every strength at one, interpolation is the exact all-constraint weighted sum for the computed distances.

**Disconnected topology:** distances to other connected components are infinite and never supply finite-distance influence. A component with no local constraint inherits the first stroke's direction as a coverage seed. Once painted locally, it uses only its own constraints; a partial-strength first local stroke blends against its existing seed. This explicitly defines full coverage without inventing paths between disconnected surfaces.

Directions are expressed in the active UV map's tangent coordinates. UV islands and triangle/vertex tangent frames are prepared once, with mirrored UV handedness preserved. A shared mesh vertex has only one `POINT` value, so UV seam corners cannot store independent directions. The dominant-area island provides that vertex's arrow frame. Split mesh vertices before painting if independent seam values are required; the split also affects surface connectivity. Equal UV-space directions need not point in the same world-space direction across differently oriented islands.

## Geodesics, preview, and scaling

The primary solver implements the **heat method of Crane, Weischedel, and Wardetzky**, from [*Geodesics in Heat* / *The Heat Method for Distance Computation*](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/). Triangle cotangent FEM stiffness, lumped mass, and the mesh adjacency are assembled on entry. SciPy sparse LU factors the heat system and an anchored Poisson system once; subsequent source queries reuse those factors. The heat time is mean edge length squared per connected component. Distance follows connected surface topology around holes and concavities, rather than Euclidean distance across space.

Heat distances are numerical approximations. This implementation uses the ordinary cotangent FEM, without intrinsic Delaunay retriangulation or a tufted Laplacian, so poor triangles can affect accuracy. If SciPy is unavailable, factorization fails, a source solve degenerates, diffusion underflows on a very long thin surface, or a component contains wire/degenerate bridges, the solver exposes a **Dijkstra edge-distance approximation** in its backend/status. Its cached adjacency respects topology, but distance has tessellation and edge-direction bias and can be slower. The paint entry checks still reject zero-area faces and collapsed UV triangles.

The influence kernel is `(scale / (distance + scale)) ** sharpness`, with mean edge length as its scale. All reachable constraints participate. Cached weighted vector sums make each new constraint blend **O(V) in time and memory**, independent of the number of past strokes. There is no per-preview `vertices × constraints` loop or KD-tree truncation of distant constraints. A BVH accelerates surface hits and preview transfers. Source-distance arrays use a **128 MiB LRU cache**; cache misses reuse the existing factorization. Rebuilding a long history can still require one solve/blend per stroke.

Editing an earlier stroke replays the later strokes to preserve their fractional-strength semantics. Its prefix is cached at the start of the gesture; later source distances are retained for the gesture and reused. Tip movement needs no new source solve, while moving the start solves its new source on the preview timer, reusing existing factorizations. Editing long histories can therefore cost O(V × later strokes) per preview, and retained edit-distance arrays can temporarily exceed the idle 128 MiB cache budget. Dense edits use the same proxy/full-resolution commit split as new strokes.

The source stays at the drag's start point, so a distance solve is needed **once on mouse-down**; mouse movement changes only its direction and the O(V) blend. The live line indicator updates on every mouse event. Field/color preview updates coalesce on a **30 Hz modal timer**, and arrow drawing uses batched GPU geometry with a default maximum of 1,500 sampled arrows. Flowmap Colors uses Blender's built-in solid vertex-color display and restores the prior shading on exit.

Above **40,000 vertices**, or when the entry benchmark source solve exceeds **33 ms**, Warpflow prepares a collapsed preview with a default target of about **5,000 vertices**. Blender's native **Decimate COLLAPSE** runs separately per connected component. Cached barycentric transfer upsamples its field to the original vertices. The face-based collapse ratio makes the vertex target approximate.

Proxy validation checks connectivity, Euler characteristic, hole boundary count, orientation, and local transfer continuity. Source lookup is restricted to the validated local surface neighborhood. Components that cannot be safely simplified retain their full geometry; if nothing can be reduced, preview uses the original solver. The sidebar identifies retained regions. **Mouse release always commits a full-resolution solve on the original mesh**, so the approximate proxy result is not persisted as the completed stroke.

Meshes containing wire edges use full-resolution preview: face decimation cannot reliably preserve the endpoints of loose-edge bridges. This explicit fallback preserves their graph connectivity throughout a stroke.

Sparse factorization memory grows with mesh size and fill-in; the 128 MiB limit covers distance caches, not the sparse factors or all Blender memory. Proxy preview reduces drag cost but does not remove original-mesh setup or release cost. Dense or poorly shaped meshes can still pause during those steps.

### Measured performance

The recorded numerical benchmark used Windows 11, an AMD Ryzen 9 9950X, Blender 5.2.1 LTS, Python 3.13.13, NumPy 2.3.4, and vendored SciPy 1.16.2. Seven source queries were measured on each regular planar grid; the table reports medians. Full data and methodology are in [`tests/benchmark_results.json`](tests/benchmark_results.json).

| Vertices | Solver setup | New heat source | Preview blend |
| ---: | ---: | ---: | ---: |
| 2,601 | 31 ms | 0.83 ms | 0.33 ms |
| 10,201 | 117 ms | 3.59 ms | 1.24 ms |
| 40,401 | 562 ms | 15.36 ms | 4.79 ms |
| 100,489 | 1,594 ms | 40.39 ms | 11.94 ms |

Separate Blender proxy measurements reduced 10,201 → 2,541 vertices in **0.306 s**, with **1.27 ms** median upsampling; 40,401 → 5,080 vertices took **1.337 s**, with **5.29 ms** upsampling. A 40,401-vertex synthetic blend took about **4 ms** with either 1 or 100 stored constraints. Run [`tests/benchmark_proxy_blender.py`](tests/benchmark_proxy_blender.py) to reproduce those setup/transfer measurements.

These are **numerical timings, not tested viewport FPS**. They exclude color-attribute uploads, GPU drawing, input handling, and baking; solver measurements also exclude UV and proxy preparation. Grids do not establish a safe upper limit for arbitrary production meshes. Meshes beyond the measured range are unvalidated; use the proxy controls and the sidebar's observed setup/solve/preview timings to assess your scene.

An integrated Blender session test also prepared a 42,025-vertex mesh with a 5,081-vertex proxy in **2.405 s**. Its preview, including the original color-attribute upload, took **5.80 ms**, excluding GPU drawing. The test verifies that release commits the full-resolution result, cancellation restores the previous colors, and repeated drag updates do not solve new distances or refactor matrices.

## Validation and v1 scope

The repository includes numerical tests for surface topology, heat-factor reuse, fallback behavior, all-constraint interpolation, strength, cancellation, and coverage. Blender tests cover UV frames, native baking and restoration, session transactions, and proxy reduction on planar, curved, holed, and coincident disconnected meshes. Numerical benchmarks are reproducible with:

Validation includes **83 numerical/Blender tests**, plus real-window workflows covering GPU drawing, native simulated mouse strokes, both endpoint handles, selection/deletion, Mirror X from either side, paired editing and undo, cancellation, persistent guides after exit, re-entry, and PNG export. The delivered ZIP is independently extracted to verify its bundled SciPy imports, heat factorizations, and register/unregister cycle in Blender 5.2.1.

```sh
blender --background --factory-startup --python scripts/benchmark.py -- --output tests/benchmark_results.json --require-vendored-scipy
blender --background --factory-startup --python-exit-code 1 --python tests/test_proxy_blender.py
```

V1 handles one active mesh per session, one point per stroke, and one PNG UV tile. Multi-object batch painting, live UV-island re-detection, path-sampled strokes, non-PNG output, and UDIM export are outside this version's scope.

## Build and test

From the source directory, using a Python environment with NumPy and SciPy:

```sh
python scripts/vendor_scipy.py
python scripts/build.py
python scripts/test.py --blender "path/to/blender.exe" --ui
```

The wheel downloader verifies pinned SHA-256 hashes. The builder includes the add-on, this guide, the GPL license and third-party notices, and produces a ZIP plus its checksum in `dist/`. It excludes Python cache files. No Blender preferences or installed add-ons are changed by the build.

The optional UI test starts a separate factory-startup Blender process and uses Blender's native event simulation for strokes, live preview, undo/redo, cancellation, exit and PNG export. It does not send OS mouse/keyboard input. Results and a viewport screenshot are written under `tests/`. See `LICENSE` and `THIRD_PARTY.md` for redistribution terms.
