"""
threedb.controls.blender.scale
==============================

[TODO]
"""

from threedb.controls.base_control import BaseControl

class ObjScaleControl(BaseControl):
    """This control scales the object

    Continuous Parameters
    ---------------------
    factor
        Scaling factor which takes any positive number. Setting the 
        factor to 1 maintains the same object size.
    """
    kind = 'pre'

    continuous_dims = {
        'factor': (0.25, 1),
    }

    discrete_dims = {}

    def apply(self, context, factor):
        """scales the object in the scene context by a factor `factor`

        Parameters
        ----------
        context
            The scene context object
        factor
            Scaling factor which takes any positive number. Setting the 
            factor to 1 maintains the same object size.
        """        
        self.ob = context['object']
        self.ob.scale = (factor,) * 3

    def unapply(self, context):
        """Rescales the object to its original dimensions

        Parameters
        ----------
        context
            The scene context object
        """
        self.ob.scale = (1.,) * 3

BlenderControl = ObjScaleControl
