"""Transactional Cycles emission baking of Warpflow's POINT color attribute.

Cycles supplies Blender's native triangulation, UV rasterization, and padding.
Compared with a manual rasterizer this adds renderer startup cost, but agrees
with Blender's material attribute interpolation and bake conventions. A private
scene, object, mesh, material, and image isolate *all* artist state. Modifiers
are intentionally excluded: the source of truth is the painted base mesh.
"""

import os
import tempfile

import bpy
import numpy as np

from .uv import build_uv_data


ATTRIBUTE_NAME = "Warpflow"
NEUTRAL = (0.5, 0.5, 0.0, 1.0)


def _normalize_pixels(pixels, blue=0.0):
    """Re-normalize interpolated RG; preserve neutral and fixed blue/alpha.

    Opposing vertex vectors can cancel inside a triangle. Exact cancellation
    has no defined direction and remains neutral rather than inventing one.
    The float tolerance also avoids magnifying renderer roundoff at neutral.
    """
    direction = pixels[:, :2] * 2.0 - 1.0
    length = np.linalg.norm(direction, axis=1)
    valid = length > 1e-6
    pixels[:, :2] = 0.5
    pixels[valid, :2] = direction[valid] / length[valid, None] * 0.5 + 0.5
    pixels[:, 2] = blue
    pixels[:, 3] = 1.0


def bake_flowmap(context, obj, filepath, resolution, margin=8):
    """Bake a raw RG flow PNG and return its absolute path.

    The active UV layer must fit the standard [0, 1] tile; UDIMs are outside
    v1's scope. Overlapping UVs remain the artist's responsibility, as with a
    native bake. PNG quantization introduces the usual small unit-length error.
    The target file is atomically replaced only after a successful image save.
    No artist materials, render/color settings, image nodes, modes, visibility,
    selection, or active object are changed, even when baking raises an error.
    """
    if obj is None or obj.type != "MESH":
        raise ValueError("Select a mesh to export its Warpflow attribute.")
    if obj.mode != "OBJECT":
        raise ValueError("Switch to Object Mode before exporting a flowmap.")
    blue = float(getattr(getattr(obj.data, 'warpflow', None), 'blue_channel', '0'))
    if isinstance(resolution, bool) or int(resolution) != resolution or not 1 <= int(resolution) <= 16384:
        raise ValueError("Flowmap resolution must be a whole number from 1 to 16384.")
    if isinstance(margin, bool) or int(margin) != margin or not 0 <= int(margin) <= 256:
        raise ValueError("Bake margin must be a whole number from 0 to 256 pixels.")
    resolution, margin = int(resolution), int(margin)
    uv_data = build_uv_data(obj.data)
    if np.any(uv_data.loop_uvs < -1e-6) or np.any(uv_data.loop_uvs > 1.0 + 1e-6):
        raise ValueError("Export supports one PNG tile. Pack the active UV map into the [0, 1] square; UDIM/out-of-tile UVs are unsupported.")
    attribute = obj.data.color_attributes.get(ATTRIBUTE_NAME)
    if attribute is None:
        raise ValueError("This mesh has no Warpflow colors. Enter Flow Paint Mode first.")
    if attribute.domain != "POINT" or attribute.data_type != "FLOAT_COLOR":
        raise ValueError("Warpflow must be a POINT / FLOAT_COLOR color attribute. Rename the conflicting attribute, then enter Flow Paint Mode.")
    source_colors = np.empty(len(obj.data.vertices) * 4, dtype=np.float32)
    attribute.data.foreach_get("color", source_colors)
    source_colors = source_colors.reshape((-1, 4))
    if not np.all(np.isfinite(source_colors)):
        raise ValueError("Warpflow colors contain non-finite values. Clear and repaint the flow field.")
    if not filepath or not str(filepath).strip():
        raise ValueError("Choose a PNG output filepath.")
    output = os.path.abspath(bpy.path.abspath(os.fspath(filepath)))
    if not output.lower().endswith(".png"):
        output += ".png"
    directory = os.path.dirname(output)
    if not os.path.isdir(directory):
        raise ValueError(f"The output folder does not exist: {directory}")
    if os.path.isdir(output):
        raise ValueError("Choose a file path, not a folder, for the flowmap PNG.")

    bake_scene = bake_mesh = bake_object = material = image = None
    temporary_path = None
    try:
        bake_scene = bpy.data.scenes.new("_Warpflow_Bake")
        bake_scene.render.engine = "CYCLES"
        bake_scene.cycles.device = "CPU"
        bake_scene.cycles.samples = 1
        bake_scene.render.bake.use_selected_to_active = False
        bake_scene.render.bake.use_clear = False
        bake_scene.render.bake.target = "IMAGE_TEXTURES"
        bake_scene.render.bake.margin = margin
        bake_scene.render.bake.margin_type = "EXTEND"
        bake_scene.render.image_settings.file_format = "PNG"
        bake_scene.render.image_settings.color_mode = "RGBA"
        bake_scene.render.image_settings.color_depth = "16"
        # Raw is an identity view, so save_render writes data channels unchanged.
        # These are settings of our disposable scene, never the artist's scene.
        bake_scene.view_settings.view_transform = "Raw"
        bake_scene.view_settings.look = "None"
        bake_scene.view_settings.exposure = 0.0
        bake_scene.view_settings.gamma = 1.0
        bake_scene.view_settings.use_curve_mapping = False
        bake_scene.render.dither_intensity = 0.0

        bake_mesh = obj.data.copy()
        bake_mesh.materials.clear()
        bake_object = bpy.data.objects.new("_Warpflow_Bake", bake_mesh)
        bake_object.matrix_world = obj.matrix_world.copy()
        bake_scene.collection.objects.link(bake_object)
        bake_view_layer = bake_scene.view_layers[0]
        bake_view_layer.objects.active = bake_object
        bake_object.select_set(True, view_layer=bake_view_layer)
        for layer in bake_mesh.uv_layers:
            layer.active_render = layer.name == uv_data.uv_layer_name
        # Preserve the artist's POINT source; only sanitize the temporary copy.
        _normalize_pixels(source_colors, blue)
        bake_mesh.color_attributes[ATTRIBUTE_NAME].data.foreach_set("color", source_colors.ravel())
        bake_mesh.update()

        image = bpy.data.images.new("_Warpflow_Bake", width=resolution, height=resolution,
                                    alpha=True, float_buffer=True, is_data=True)
        image.colorspace_settings.name = "Non-Color"
        image.alpha_mode = "CHANNEL_PACKED"
        image.use_view_as_render = False
        pixels = np.empty((resolution * resolution, 4), dtype=np.float32)
        pixels[:] = NEUTRAL
        pixels[:, 2] = blue
        image.pixels.foreach_set(pixels.ravel())
        image.update()
        material = bpy.data.materials.new("_Warpflow_Bake")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        colors = nodes.new("ShaderNodeVertexColor")
        colors.layer_name = ATTRIBUTE_NAME
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        output_node = nodes.new("ShaderNodeOutputMaterial")
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        for node in nodes:
            node.select = False
        image_node.select = True
        nodes.active = image_node
        links = material.node_tree.links
        links.new(colors.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs[0], output_node.inputs["Surface"])
        bake_mesh.materials.append(material)
        bake_mesh.polygons.foreach_set("material_index", np.zeros(len(bake_mesh.polygons), dtype=np.int32))
        bake_view_layer.update()
        # Context overrides are restored by Blender even on operator failure.
        # The window itself never switches scenes; original selection is intact.
        with context.temp_override(scene=bake_scene, view_layer=bake_view_layer,
                                   object=bake_object, active_object=bake_object,
                                   selected_objects=[bake_object],
                                   selected_editable_objects=[bake_object]):
            result = bpy.ops.object.bake(type="EMIT", target="IMAGE_TEXTURES",
                                         margin=margin, margin_type="EXTEND",
                                         use_clear=False, use_selected_to_active=False,
                                         uv_layer=uv_data.uv_layer_name)
            if "FINISHED" not in result:
                raise RuntimeError("Cycles did not finish the flowmap bake.")
        image.pixels.foreach_get(pixels.ravel())
        if not np.all(np.isfinite(pixels)):
            raise RuntimeError("Cycles returned invalid flowmap pixels.")
        _normalize_pixels(pixels, blue)
        image.pixels.foreach_set(pixels.ravel())
        image.update()
        descriptor, temporary_path = tempfile.mkstemp(prefix=".warpflow-", suffix=".png", dir=directory)
        os.close(descriptor)
        image.save_render(temporary_path, scene=bake_scene)
        if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
            raise RuntimeError("Blender did not write the flowmap PNG.")
        os.replace(temporary_path, output)
        temporary_path = None
        return output
    finally:
        # Dispose every private datablock on both success and exceptions. Nothing
        # in this cleanup touches the original object's mesh or materials.
        if bake_object is not None:
            bpy.data.objects.remove(bake_object, do_unlink=True)
        if bake_scene is not None:
            bpy.data.scenes.remove(bake_scene, do_unlink=True)
        if bake_mesh is not None:
            bpy.data.meshes.remove(bake_mesh, do_unlink=True)
        if material is not None:
            bpy.data.materials.remove(material, do_unlink=True)
        if image is not None:
            bpy.data.images.remove(image, do_unlink=True)
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
