# Third-party software

Warpflow's original Python code is licensed under GNU GPL v3 or later; see LICENSE.

The Windows ZIP redistributes the official **SciPy 1.16.2** CPython 3.11 and 3.13
Windows x86-64 wheels. SciPy is BSD licensed. Its complete license, copyright,
and bundled numerical-library notices (including OpenBLAS/LAPACK) are preserved
in each `_vendor/win_amd64/cp*/scipy-1.16.2.dist-info/` directory. These bundled
dependencies retain their own licenses.

Wheel URLs and SHA-256 hashes are pinned in `scripts/vendor_scipy.py` in the
source project. The script downloads and extracts wheels into the add-on only;
it does not run pip, add dependencies to Blender, or replace NumPy.

Release source: https://pypi.org/project/scipy/1.16.2/

Blender provides NumPy, mathutils, gpu, bpy and the native Cycles bake/Decimate
facilities. They are not redistributed as part of this add-on.
