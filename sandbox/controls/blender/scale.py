from sandbox.controls.base_control import BaseControl

class ObjScaleControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'factor': (0.25, 1),
    }

    discrete_dims = {}

    def apply(self, context, factor):
        self.ob = context['object']
        self.ob.scale = (factor,) * 3

    def unapply(self, context):
        self.ob.scale = (1.,) * 3

BlenderControl = ObjScaleControl
