import numpy as np
from sandbox.controls.base_control import BaseControl
from sandbox.controls.blender.blender_utils import post_translate

class PositionControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'offset_X': (-1, 1),
        'offset_Y': (-1, 1),
        'offset_Z': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, offset_X, offset_Y, offset_Z):

        ob = context['object']
        post_translate(ob, [offset_X, offset_Y, offset_Z])


BlenderControl = PositionControl
