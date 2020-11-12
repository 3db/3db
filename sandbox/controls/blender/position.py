import numpy as np
from sandbox.controls.base_control import BaseControl
try:
    from sandbox.controls.blender.blender_utils import (
        post_translate,
        cleanup_translate_containers)
except:
    pass

class PositionControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'offset_X': (-1, 1),
        'offset_Y': (-1, 1),
        'offset_Z': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, offset_X, offset_Y, offset_Z):
        from mathutils import Vector

        ob = context['object']
        self.ob = ob
        post_translate(ob, Vector([offset_X, offset_Y, offset_Z]))

    def unapply(self):
        cleanup_translate_containers(self.ob)


BlenderControl = PositionControl
