import numpy as np
from sandbox.controls.base_control import BaseControl

class MugLiquidController(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'ratio_milk': (0, 1),
        'ratio_coffee': (0, 1),
        'ratio_water': (0, 1),
    }

    discrete_dims = {}

    def apply(self, context, ratio_milk, ratio_coffee, ratio_water):
        import bpy
        import mathutils

        tot = ratio_milk + ratio_coffee + ratio_water

        # eul = mathutils.Euler((rotation_X, rotation_Y, rotation_Z), 'XYZ')
        ob = context['object']
        material = ob.material_slots['liquid'].material.node_tree
        material.nodes['milk_coffe_ratio'].inputs['Fac'].default_value = ratio_milk
        material.nodes['water_ratio'].inputs['Fac'].default_value = ratio_water

BlenderControl = MugLiquidController
