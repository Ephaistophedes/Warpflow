"""Saved constraints are undoable metadata; Warpflow color data is the field."""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, IntVectorProperty,
                       PointerProperty, StringProperty, EnumProperty)


def settings_changed(self, context):
    from . import session
    if session.ACTIVE is not None:
        session.ACTIVE.settings_dirty = True
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class WARPFLOW_PG_stroke(bpy.types.PropertyGroup):
    vertices: IntVectorProperty(size=3, min=0)
    barycentric: FloatVectorProperty(size=3)
    tip_vertices: IntVectorProperty(size=3, min=0)
    tip_barycentric: FloatVectorProperty(size=3)
    has_tip: BoolProperty(default=False)
    direction: FloatVectorProperty(size=2)
    strength: FloatProperty(default=1.0, min=0.0, max=1.0)
    mirror_pair: StringProperty(default='')


class WARPFLOW_PG_mesh(bpy.types.PropertyGroup):
    strokes: CollectionProperty(type=WARPFLOW_PG_stroke)
    geometry_signature: StringProperty()


class WARPFLOW_PG_settings(bpy.types.PropertyGroup):
    mirror_x: BoolProperty(name="Mirror X", description="Mirror new strokes and linked endpoint edits across the object's local X=0 plane, from either side", default=False, update=settings_changed)
    sharpness: FloatProperty(name="Influence Sharpness", description="Inverse surface-distance exponent; higher values give tighter transitions", default=2.0, min=0.25, max=8.0, update=settings_changed)
    strength: FloatProperty(name="Strength", description="Blend the pre-stroke field toward this stroke's result, then normalize", default=1.0, min=0.0, max=1.0)
    show_stroke: BoolProperty(name="In-progress Stroke", default=True)
    show_strokes: BoolProperty(name="Saved Stroke Arrows", description="Show completed arrows; select and drag their endpoints in Flow Paint Mode", default=True, update=settings_changed)
    show_arrows: BoolProperty(name="Direction Arrows", default=True)
    show_preview: BoolProperty(name="Flowmap Colors", default=True, update=settings_changed)
    arrow_count: IntProperty(name="Maximum Arrows", default=1500, min=50, max=10000, update=settings_changed)
    arrow_scale: FloatProperty(name="Arrow Size", default=1.0, min=0.1, max=10.0, update=settings_changed)
    proxy_threshold: IntProperty(name="Proxy Above Vertices", description="Prepare a collapsed mesh for interactive preview above this vertex count", default=40000, min=1000, max=1000000)
    proxy_target: IntProperty(name="Proxy Target Vertices", default=5000, min=200, max=50000)
    resolution: EnumProperty(name="Resolution", items=[('512', '512', ''), ('1024', '1024', ''), ('2048', '2048', ''), ('CUSTOM', 'Custom', '')], default='1024')
    custom_resolution: IntProperty(name="Custom Resolution", default=2048, min=64, max=8192)
    bake_margin: IntProperty(name="Padding", default=8, min=0, max=64)


CLASSES = (WARPFLOW_PG_stroke, WARPFLOW_PG_mesh, WARPFLOW_PG_settings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Mesh.warpflow = PointerProperty(type=WARPFLOW_PG_mesh)
    bpy.types.Scene.warpflow = PointerProperty(type=WARPFLOW_PG_settings)


def unregister():
    del bpy.types.Scene.warpflow
    del bpy.types.Mesh.warpflow
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
