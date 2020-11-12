import numpy as np
from sandbox.controls.base_control import BaseControl

TRANSLATE_PREFIX = 'translation_control_'

class PositionControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'offset_X': (-1, 1),
        'offset_Y': (-1, 1),
        'offset_Z': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, offset_X, offset_Y, offset_Z):
        import bpy
        import mathutils

        def post_translate(object, offset):
            if (object.parent is None
                    or not object.parent.name.startswith(TRANSLATE_PREFIX)):
                bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD',
                                         location=(0, 0, 0), scale=(1, 1, 1))
                container = bpy.context.object
                container.name = TRANSLATE_PREFIX + object.name
                object.parent = container
            object.parent.location = offset

        ob = context['object']
        post_translate(ob, [offset_X, offset_Y, offset_Z])

BlenderControl = PositionControl
