from .base_control import BaseControl

class ObjScaleControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'scale': (0.25, 1),
    }

    discrete_dims = {}

    def apply(self, context, scale):
        self.ob = context['object']
        self.ob.scale = (scale,) * 3

    def unapply(self):
        self.ob.scale = (1.,) * 3


Control = ObjScaleControl

