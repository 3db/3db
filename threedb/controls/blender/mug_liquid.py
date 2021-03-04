import numpy as np
from threedb.controls.base_control import BaseControl

class MugLiquidController(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'ratio_milk': (0, 1),
        'ratio_coffee': (0, 1),
    }

    discrete_dims = {}

    def apply(self, context, ratio_milk, ratio_coffee):
        import bpy
        import mathutils

        ratio_water = 1 - ratio_coffee - ratio_milk

        ob = context['object']
        material = ob.material_slots['liquid'].material.node_tree
        material.nodes["coffee_milk_ratio"].outputs[0].default_value = ratio_coffee / (ratio_coffee + ratio_milk)
        material.nodes["water_ratio"].outputs[0].default_value = ratio_water

BlenderControl = MugLiquidController
