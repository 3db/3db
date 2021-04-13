"""
threedb.controls.blender.mug_liquid
===================================

Defines the MugLiquidControl
"""

from ...try_bpy import bpy
import mathutils
from typing import Any, Dict, Tuple
from ..base_control import PreProcessControl


class MugLiquidControl(PreProcessControl):
    """Change the material of the liquid present in the mug object

    Note
    ----
    This control assumes that the surface of the liquid has a manterial named
    "liquid" and that the object we want to modify is the target of this render.

    The ratio of water will be 1 - ratio_milk - ratio_coffee

    Continuous Dimensions:

    - ``ratio_milk`` (float): The ratio of milk (range: [0, 1])
    - ``ratio_coffee`` (float): The ratio of coffee (range: [0, 1])
    """
    @property
    def continuous_dims(self) -> Dict[str, Tuple[float, float]]:
        return {
            'ratio_milk': (0, 1),
            'ratio_coffee': (0, 1),
        }

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Change the material of the liquid

        Parameters
        ----------
        context : Dict[str, Any]
            The scene context
        control_args : Dict[str, Any]
            The arguments for this control, should have keys ``ratio_milk`` and
            ``ratio_coffee`` (see the class docstring for information).
        """
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        ratio_coffee: float
        ratio_milk: float
        ratio_coffee, ratio_milk = control_args['ratio_coffee'], control_args['ratio_milk']

        ratio_water = 1 - ratio_coffee - ratio_milk

        obj = context['object']
        material = obj.material_slots['liquid'].material.node_tree
        material.nodes["coffee_milk_ratio"].outputs[0].default_value = (
            ratio_coffee / (ratio_coffee + ratio_milk))
        material.nodes["water_ratio"].outputs[0].default_value = ratio_water


Control = MugLiquidControl
