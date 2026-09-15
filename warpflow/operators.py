"""Enter precomputes; a lightweight controller owns events and native undo steps."""
import time
import traceback

import bpy
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper
from bpy_extras import view3d_utils

from . import session
from .drawing import FlowDrawing
from .guides import pick_stroke


class WARPFLOW_OT_enter(bpy.types.Operator):
    bl_idname = 'warpflow.enter'
    bl_label = 'Enter Flow Paint Mode'
    bl_description = 'Prepare surface distances, UV frames and preview before painting'

    @classmethod
    def poll(cls, context):
        return (session.ACTIVE is None and context.area is not None
                and context.area.type == 'VIEW_3D' and context.object is not None
                and context.object.type == 'MESH' and context.object.mode == 'OBJECT')

    def execute(self, context):
        wm = context.window_manager
        wm.progress_begin(0, 100)
        context.window.cursor_set('WAIT')
        context.area.header_text_set('Warpflow: preparing surface solver and UV frames...')
        try:
            prepared = session.PaintSession(context, progress=lambda v: wm.progress_update(100 * v))
            session.ACTIVE = prepared
            # A baseline is needed before the first manual per-stroke undo push.
            # The long-lived controller is NOT an UNDO operator: otherwise a
            # second session-wide undo snapshot is created when Esc finishes it.
            bpy.ops.ed.undo_push(message='Enter Warpflow')
            bpy.ops.warpflow.paint('INVOKE_DEFAULT')
            backend = prepared.solver.backend
            self.report({'INFO'}, f'Warpflow ready: {len(prepared.vertices):,} vertices, {backend}, {prepared.setup_seconds:.2f}s setup')
            if prepared.uv.warnings:
                self.report({'WARNING'}, prepared.uv.warnings[0])
            return {'FINISHED'}
        except Exception as exc:
            session.end_session(context)
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            wm.progress_end()
            context.window.cursor_set('DEFAULT')
            if session.ACTIVE is None:
                context.area.header_text_set(None)


class WARPFLOW_OT_paint(bpy.types.Operator):
    bl_idname = 'warpflow.paint'
    bl_label = 'Paint Warpflow'
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        if session.ACTIVE is None:
            return {'CANCELLED'}
        # All expensive geometric/numerical setup is already complete.
        self.paint_session = session.ACTIVE
        s = self.paint_session
        s.controller = self
        self.last_tick = 0.0
        self.selecting = False
        s.draw = FlowDrawing(s)
        s.draw.install()
        s.update_preview_shading()
        s.timer = context.window_manager.event_timer_add(1.0 / 30.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('PAINT_BRUSH')
        self._header()
        return {'RUNNING_MODAL'}

    def _header(self):
        s = self.paint_session
        if s.area:
            suffix = f' | Proxy: {len(s.proxy.vertices):,} vertices' if s.proxy is not None else ''
            selected = f' | Selected stroke {s.selected_stroke + 1}' if s.selected_stroke >= 0 else ''
            s.area.header_text_set(f'Warpflow | Drag to paint | Click arrow / drag handles to edit | Del: delete | Ctrl-drag: new | Esc: exit{selected}{suffix}')

    def _mouse(self, event):
        r = self.paint_session.region
        return event.mouse_x - r.x, event.mouse_y - r.y

    def _inside(self, event):
        s = self.paint_session
        r = s.region
        if not (r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height):
            return False
        # Sidebars can overlap WINDOW; leave their controls fully interactive.
        for other in s.area.regions:
            if other.type in {'UI', 'TOOLS', 'HEADER', 'TOOL_HEADER'} and other.width > 1:
                if other.x <= event.mouse_x < other.x + other.width and other.y <= event.mouse_y < other.y + other.height:
                    return False
        return True

    def _ray(self, mouse):
        s = self.paint_session
        rv3d = s.area.spaces.active.region_3d
        return (view3d_utils.region_2d_to_origin_3d(s.region, rv3d, mouse),
                view3d_utils.region_2d_to_vector_3d(s.region, rv3d, mouse))

    def _update_drag(self, event):
        s = self.paint_session
        if not s.drag:
            return
        mouse = self._mouse(event)
        # This cheap screen indicator is redrawn for every modal mouse event.
        s.area.tag_redraw()
        if np.linalg.norm(np.array(mouse) - np.array(s.drag['mouse_start'])) < 3.0:
            # Returning to the start removes the gesture's direction. Restore
            # once when crossing back into this dead zone, keeping the original
            # source alive so dragging out again can continue the same gesture.
            had_direction = s.drag['direction'] is not None
            s.drag['direction'] = None
            s.drag['mouse_end'] = mouse
            s.drag['pending'] = False
            if had_direction:
                session.write_field(s.obj.data, s.drag['before'])
                if s.draw:
                    s.draw.update_arrows(s.drag['before'])
            return
        origin, ray = self._ray(mouse)
        hit, normal, index, distance = s.bvh.ray_cast(origin, ray)
        start = s.drag['position']
        if hit is None or s.solver.components[s.triangles[index, 0]] != s.solver.components[s.drag['vertices'][0]]:
            # Retain the last valid surface endpoint. A floating tangent-plane
            # fallback would change the saved arrow's length when it is released.
            return
        delta_local = s.inverse_matrix.to_3x3() @ (hit - start)
        direction = s.uv.direction_to_uv(s.drag['triangle'], delta_local)
        if direction is not None:
            s.drag['direction'] = direction
            s.drag['mouse_end'] = mouse
            s.set_stroke_tip(index, hit)
            s.drag['pending'] = True

    def _update_edit(self, event):
        s = self.paint_session
        mouse = self._mouse(event)
        # A handle has a generous pick radius. A click selects it without
        # snapping the saved anchor to the slightly offset click location.
        if not s.edit['changed'] and np.linalg.norm(np.array(mouse) - np.array(s.edit['mouse_start'])) < 3.0:
            return
        origin, ray = self._ray(mouse)
        hit, normal, index, distance = s.bvh.ray_cast(origin, ray)
        # Existing endpoints remain attached to the original connected surface.
        # Off-mesh motion retains the last valid position until the ray returns.
        if hit is not None:
            stroke = s.obj.data.warpflow.strokes[s.edit['index']]
            if s.solver.components[s.triangles[index, 0]] == s.solver.components[stroke.vertices[0]]:
                s.update_edit(index, hit, mouse)
        s.area.tag_redraw()

    def modal(self, context, event):
        s = self.paint_session
        if s.ended or session.ACTIVE is not s:
            return {'FINISHED'}
        try:
            if (not s.obj or context.area != s.area or s.area.type != 'VIEW_3D'
                    or s.obj.mode != 'OBJECT' or s.obj.matrix_world != s.matrix):
                session.end_session(context)
                return {'FINISHED'}
            if event.type == 'ESC' and event.value == 'PRESS':
                if s.edit is not None:
                    s.cancel_edit()
                    self._header()
                    s.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                session.end_session(context)
                return {'FINISHED'}
            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                s.cancel_edit()
                s.cancel_stroke()
                self.selecting = False
                s.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'Z' and event.value == 'PRESS' and (event.ctrl or event.oskey):
                s.cancel_edit()
                s.cancel_stroke()
                with context.temp_override(area=s.area, region=s.region):
                    operator = bpy.ops.ed.redo if event.shift else bpy.ops.ed.undo
                    if operator.poll():
                        operator()
                if not s.ended:
                    self._header()
                return {'RUNNING_MODAL'}
            if event.type == 'TIMER':
                # Blender 5.x Event exposes no timer handle. Other add-ons may
                # emit timers too, so use elapsed time to cap field uploads.
                now = time.perf_counter()
                if now - self.last_tick < 1.0 / 30.0:
                    return {'PASS_THROUGH'}
                self.last_tick = now
                if len(s.obj.data.vertices) != len(s.vertices):
                    self.report({'WARNING'}, 'Mesh topology changed; Warpflow session ended.')
                    session.end_session(context)
                    return {'FINISHED'}
                if s.settings_dirty:
                    if s.accumulator.sharpness != s.settings.sharpness:
                        s.cancel_edit()
                        s.cancel_stroke()
                        s.rebuild()
                        bpy.ops.ed.undo_push(message='Warpflow Influence Sharpness')
                    else:
                        s.draw.update_arrows(session.read_field(s.obj.data))
                    s.settings_dirty = False
                    s.update_preview_shading()
                    s.area.tag_redraw()
                if s.edit and s.edit['pending']:
                    s.preview_edit()
                    s.area.tag_redraw()
                elif s.drag and s.drag['pending']:
                    s.preview()
                    s.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self._inside(event) and not event.alt:
                if not s.rebind():
                    self.report({'WARNING'}, 'Geometry or UVs changed. Clear Strokes and re-enter paint mode.')
                    session.end_session(context)
                    return {'FINISHED'}
                mouse = self._mouse(event)
                picked = pick_stroke(s, mouse) if s.settings.show_strokes and not event.ctrl else None
                if picked is not None:
                    index, part = picked
                    s.selected_stroke = index
                    if part in {'START', 'TIP'}:
                        s.begin_edit(index, part, mouse)
                    else:
                        self.selecting = True
                    self._header()
                    s.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                s.selected_stroke = -1
                self.selecting = False
                self._header()
                origin, ray = self._ray(mouse)
                hit, normal, index, distance = s.bvh.ray_cast(origin, ray)
                if hit is not None:
                    s.begin_stroke(index, hit, mouse)
                    return {'RUNNING_MODAL'}
            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                if s.edit:
                    self._update_edit(event)
                    return {'RUNNING_MODAL'}
                if s.drag:
                    self._update_drag(event)
                    return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and s.edit:
                self._update_edit(event)
                if s.commit_edit():
                    bpy.ops.ed.undo_push(message='Warpflow Edit Stroke')
                self._header()
                s.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.selecting:
                self.selecting = False
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and s.drag:
                self._update_drag(event)
                if s.commit_stroke():
                    # Exactly one native undo boundary per completed stroke.
                    # Preview/cancel/exit never push undo state.
                    bpy.ops.ed.undo_push(message='Warpflow Stroke')
                self._header()
                s.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if self._inside(event) and event.value == 'PRESS' and event.type in {'X', 'DEL'}:
                if bpy.ops.warpflow.delete_stroke.poll():
                    bpy.ops.warpflow.delete_stroke()
                return {'RUNNING_MODAL'}
            # Orbit/pan/zoom and N-panel remain usable. Block object transforms,
            # edit-mode switches and deletion while cached surface data is active.
            if self._inside(event) and event.value == 'PRESS' and event.type in {'G', 'R', 'S', 'TAB', 'X', 'DEL'}:
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, f'Warpflow stopped: {exc}')
            session.end_session(context)
            return {'FINISHED'}

    def cancel(self, context):
        session.end_session(context)


class WARPFLOW_OT_exit(bpy.types.Operator):
    bl_idname = 'warpflow.exit'
    bl_label = 'Exit Flow Paint Mode'

    def execute(self, context):
        session.end_session(context)
        return {'FINISHED'}


class WARPFLOW_OT_delete_stroke(bpy.types.Operator):
    bl_idname = 'warpflow.delete_stroke'
    bl_label = 'Delete Selected Stroke'
    bl_description = 'Remove the selected directional constraint and update the field'

    @classmethod
    def poll(cls, context):
        s = session.ACTIVE
        return s is not None and s.obj is not None and 0 <= s.selected_stroke < len(s.obj.data.warpflow.strokes)

    def execute(self, context):
        s = session.ACTIVE
        if s is None:
            return {'CANCELLED'}
        try:
            s.cancel_edit()
            s.cancel_stroke()
            if not s.remove_selected():
                return {'CANCELLED'}
            bpy.ops.ed.undo_push(message='Warpflow Delete Stroke')
            if getattr(s, 'controller', None):
                s.controller._header()
            if s.area:
                s.area.tag_redraw()
            return {'FINISHED'}
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class WARPFLOW_OT_clear(bpy.types.Operator):
    bl_idname = 'warpflow.clear'
    bl_label = 'Clear Strokes'
    bl_description = 'Remove all constraints and reset the field to neutral (undoable)'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return session.ACTIVE is None and context.object is not None and context.object.type == 'MESH' and context.object.mode == 'OBJECT'

    def execute(self, context):
        mesh = context.object.data
        try:
            session.ensure_attribute(mesh)
            mesh.warpflow.strokes.clear()
            mesh.warpflow.geometry_signature = ''
            session.write_field(mesh, np.zeros((len(mesh.vertices), 2)))
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class WARPFLOW_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = 'warpflow.export'
    bl_label = 'Export Flowmap'
    bl_description = 'Bake the active mesh Warpflow attribute to a raw RG PNG'
    filename_ext = '.png'
    filter_glob: StringProperty(default='*.png', options={'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH')

    @classmethod
    def poll(cls, context):
        return (session.ACTIVE is None and context.object is not None
                and context.object.type == 'MESH' and context.object.mode == 'OBJECT'
                and session.ATTRIBUTE in context.object.data.color_attributes)

    def invoke(self, context, event):
        self.filepath = bpy.path.clean_name(context.object.name) + '_flow.png'
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        from .bake import bake_flowmap
        settings = context.scene.warpflow
        resolution = settings.custom_resolution if settings.resolution == 'CUSTOM' else int(settings.resolution)
        context.window_manager.progress_begin(0, 1)
        context.window.cursor_set('WAIT')
        try:
            path = bake_flowmap(context, context.object, self.filepath, resolution, settings.bake_margin)
            self.report({'INFO'}, f'Exported {resolution} x {resolution} flowmap: {path}')
            return {'FINISHED'}
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            context.window_manager.progress_end()
            context.window.cursor_set('DEFAULT')


CLASSES = (WARPFLOW_OT_enter, WARPFLOW_OT_paint, WARPFLOW_OT_exit, WARPFLOW_OT_delete_stroke, WARPFLOW_OT_clear, WARPFLOW_OT_export)
