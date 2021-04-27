"""
threedb.controls.blender.material
=================================

Change object materials. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/material.yaml>`_. 
"""

from os import path
from glob import glob
from typing import Any, Dict
from pathlib import Path

from ...try_bpy import bpy
from ..base_control import PreProcessControl

MATERIAL_FOLDER = 'blender_control_material'

class MaterialControl(PreProcessControl):
    """Control that swap material of an object with another one

    Discrete Dimensions:

    - ``replacement_material``: The name of the material that replaces the
        original material.

    .. note::
        The possible values for ``replacement_material`` can be any file name
        found in ``$BLENDER_DATA/blender_control_material``.

        Each of these file should only contain a single material.

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/material/images/image_1.png
            :width: 100
            :group: material

        .. thumbnail:: /_static/logs/material/images/image_2.png
            :width: 100
            :group: material

        .. thumbnail:: /_static/logs/material/images/image_3.png
            :width: 100
            :group: material

        .. thumbnail:: /_static/logs/material/images/image_4.png
            :width: 100
            :group: material

        .. thumbnail:: /_static/logs/material/images/image_5.png
            :width: 100
            :group: material
        
        Examples of various material substitutions.
    """
    def __init__(self, root_folder: str):
        mat_folder = Path(root_folder) / MATERIAL_FOLDER
        files_in_folder = mat_folder.glob('*.blend')
        files_in_folder = [f.name for f in files_in_folder]
        files_in_folder.append('keep_original')
        discrete_dims = {
            "replacement_material": files_in_folder
        }
        super().__init__(root_folder, discrete_dims=discrete_dims)


    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Replace the material of the target object with the material
        corresponding to ``replacement_material``.

        Parameters
        ----------
        context : Dict[str, Any]
            The blender scene context.
        control_args : Dict[str, Any]
            Must have key ``replacement_material`` containing the file name of the
            replacement material to use.
        """
        arg_check = self.check_arguments(control_args)
        assert arg_check[0], arg_check[1]

        state = {
            'added_materials': [],
            'replaced_materials': []
        }

        context['material_control_state'] = state

        replacement_material = control_args['replacement_material']
        if replacement_material == "keep_original":
            return

        fname = path.join(self.root_folder, MATERIAL_FOLDER, replacement_material)
        current_materials = set(x.name for x in bpy.data.materials)

        with bpy.data.libraries.load(fname) as (data_from, data_to):
            for material in data_from.materials:
                if material not in current_materials:
                    state['added_materials'].append(material)
                    data_to.materials.append(material)

        if len(state['added_materials']) != 1:
            raise Exception(f"File {fname} contains more than one material")

        replacing_with = state['added_materials'][0]

        for slot in context['object'].material_slots:
            try:
                state['replaced_materials'].append(slot.material.name)
            except Exception:
                state['replaced_materials'].append(None)

            slot.material = bpy.data.materials[replacing_with]


    def unapply(self, context: Dict[str, Any]) -> None:
        state = context['material_control_state']

        for slot, material_name in zip(context['object'].material_slots,
                                       state['replaced_materials']):

            if material_name is None:
                slot.material = None
            else:
                slot.material = bpy.data.materials[material_name]

        for material_name in state['added_materials']:
            bpy.data.materials.remove(bpy.data.materials[material_name])

Control = MaterialControl
