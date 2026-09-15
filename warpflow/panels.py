import bpy
import textwrap
from . import session


class WARPFLOW_PT_paint(bpy.types.Panel):
    bl_label = 'Warpflow'
    bl_idname = 'WARPFLOW_PT_paint'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Warpflow'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.warpflow
        active = session.ACTIVE
        obj = context.object
        if active:
            layout.operator('warpflow.exit', icon='PAUSE')
            layout.label(text='Drag LMB to paint. Esc to exit.')
            layout.label(text='Click a stroke; drag either handle.')
            layout.label(text='Ctrl Z: undo | Shift Ctrl Z: redo')
        else:
            layout.operator('warpflow.enter', icon='BRUSH_DATA')
            if obj is None or obj.type != 'MESH':
                layout.label(text='Select one mesh in Object Mode.', icon='INFO')
            elif not obj.data.uv_layers.active:
                layout.label(text='An existing UV unwrap is required.', icon='ERROR')
        distance = layout.column(align=True)
        distance.enabled = not bool(active)
        distance.prop(settings, 'distance_mode')
        if settings.distance_mode == 'VOLUME':
            distance.prop(settings, 'volume_resolution')
            if not active:
                layout.label(text='Volume requires a closed manifold mesh.', icon='INFO')
        layout.prop(settings, 'sharpness')
        layout.prop(settings, 'strength', slider=True)
        layout.prop(settings, 'mirror_x', toggle=True, icon='MOD_MIRROR')
        column = layout.column(align=True)
        column.prop(settings, 'show_stroke')
        column.prop(settings, 'show_strokes')
        column.prop(settings, 'show_arrows')
        column.prop(settings, 'show_preview')
        if obj and obj.type == 'MESH':
            layout.label(text=f'{len(obj.data.warpflow.strokes)} strokes | {len(obj.data.vertices):,} vertices')
        layout.operator('warpflow.clear', icon='TRASH')
        if active:
            if active.selected_stroke >= 0:
                layout.label(text=f'Selected stroke: {active.selected_stroke + 1}')
                if settings.mirror_x and active._partner(active.selected_stroke) is not None:
                    layout.label(text='Mirrored pair selected')
            layout.operator('warpflow.delete_stroke', icon='X')
            box = layout.box()
            fallback = active.solver.stats.get('fallback_reason', '')
            if active.distance_mode == 'VOLUME':
                box.label(text='Volumetric interior distance field', icon='INFO')
            else:
                box.label(text='Surface heat solver' if not fallback else 'Graph distance approximation', icon='INFO')
            if fallback:
                for line in textwrap.wrap(fallback, width=max(24, context.region.width // 7 - 6)):
                    box.label(text=line)
            box.label(text=f'Setup {active.setup_seconds:.2f}s | Solve {active.solve_ms:.1f}ms')
            box.label(text=f'Last preview {active.last_preview_ms:.1f}ms')
            box.label(text=f'{active.uv.island_count} UV islands')
            if active.proxy is not None:
                box.label(text=f'Preview proxy: {len(active.proxy.vertices):,} vertices')
            if active.notice:
                for line in textwrap.wrap(active.notice, width=max(24, context.region.width // 7 - 6)):
                    box.label(text=line)


class WARPFLOW_PT_display(bpy.types.Panel):
    bl_label = 'Display & Performance'
    bl_parent_id = 'WARPFLOW_PT_paint'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Warpflow'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        settings = context.scene.warpflow
        self.layout.prop(settings, 'arrow_count')
        self.layout.prop(settings, 'arrow_scale')
        col = self.layout.column()
        col.enabled = session.ACTIVE is None
        col.prop(settings, 'proxy_threshold')
        col.prop(settings, 'proxy_target')
        col.label(text='Proxy settings apply when entering mode.')


class WARPFLOW_PT_export(bpy.types.Panel):
    bl_label = 'Export Flowmap'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Warpflow'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.warpflow
        layout.prop(settings, 'resolution')
        if settings.resolution == 'CUSTOM':
            layout.prop(settings, 'custom_resolution')
        layout.prop(settings, 'bake_margin')
        layout.operator('warpflow.export', icon='EXPORT')
        layout.label(text='PNG | RG direction | Non-Color data')
        if session.ACTIVE:
            layout.label(text='Exit paint mode before exporting.', icon='INFO')


CLASSES = (WARPFLOW_PT_paint, WARPFLOW_PT_display, WARPFLOW_PT_export)
