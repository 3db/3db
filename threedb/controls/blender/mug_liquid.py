"""Defines the MugLiquidController"""

from threedb.controls.base_control import BaseControl


class MugLiquidController(BaseControl):
    """Change the material of the liquid present in the mug object

    Note
    ----
    It assumes that the surface of the liquid has a manterial named "liquid"
    and that the object we want to modify is the target of this render

    The ratio of water will be 1 - ratio_milk - ratio_coffee

    Continuous Dimensions
    ---------------------
    ratio_milk
        The ratio of milk
    ratio_coffee
        The ratio of coffee
    """
    kind = 'pre'

    continuous_dims = {
        'ratio_milk': (0, 1),
        'ratio_coffee': (0, 1),
    }

    discrete_dims = {}

    def apply(self, context, ratio_milk, ratio_coffee):
        """Change the material of the liquid

        Parameters
        ----------
        context
            The scene context
        ratio_milk
            The ratio of milk
        ratio_coffee
            The ratio of coffee
        """
        import bpy
        import mathutils

        ratio_water = 1 - ratio_coffee - ratio_milk

        obj = context['object']
        material = obj.material_slots['liquid'].material.node_tree
        material.nodes["coffee_milk_ratio"].outputs[0].default_value = (
            ratio_coffee / (ratio_coffee + ratio_milk))
        material.nodes["water_ratio"].outputs[0].default_value = ratio_water


BlenderControl = MugLiquidController
