"""Warpflow: paint a complete, surface-weighted directional flow field."""
bl_info = {
    'name': 'Warpflow',
    'author': 'Warpflow contributors',
    'version': (1, 1, 0),
    'blender': (5, 0, 0),
    'location': '3D View > Sidebar > Warpflow',
    'description': 'Paint directional constraints and bake full-coverage surface flowmaps',
    'category': 'Paint',
}

from . import dependencies
dependencies.prepare()


def register():
    import bpy
    if bpy.app.version < (5, 0, 0):
        raise RuntimeError('Warpflow requires Blender 5.0 or later.')
    from . import properties, operators, panels, session, drawing
    properties.register()
    for cls in operators.CLASSES + panels.CLASSES:
        bpy.utils.register_class(cls)
    for handlers, callback in ((bpy.app.handlers.undo_post, session.undo_redo_post),
                               (bpy.app.handlers.redo_post, session.undo_redo_post),
                               (bpy.app.handlers.load_pre, session.load_pre)):
        if callback not in handlers:
            handlers.append(callback)
    drawing.register_guides()


def unregister():
    import bpy
    from . import properties, operators, panels, session, drawing
    session.end_session()
    drawing.unregister_guides()
    for handlers, callback in ((bpy.app.handlers.undo_post, session.undo_redo_post),
                               (bpy.app.handlers.redo_post, session.undo_redo_post),
                               (bpy.app.handlers.load_pre, session.load_pre)):
        if callback in handlers:
            handlers.remove(callback)
    for cls in reversed(operators.CLASSES + panels.CLASSES):
        bpy.utils.unregister_class(cls)
    properties.unregister()
