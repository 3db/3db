import numpy as np

class CameraControl:
    kind = 'pre'

    continuous_dims = {
        'view_point_x': (-1, 1),
        'view_point_y': (-1, 1),
        'view_point_z': (0, 1),
        'zoom_factor': (-np.pi, np.pi),
        'aperture': (1, 32),
        'focal_length': (10, 400),
    }

    discrete_dims = {}

    def apply(self, context, view_point_x, view_point_y, view_point_z,
              zoom_factor, aperture, focal_length):
        import bpy
        from mathutils import Vector
        from sandbox.rendering.utils import (sample_upper_sphere,
                                             lookat_viewport)

        zoomout_factor = 1 / zoom_factor

        camera = bpy.data.objects['Camera']
        camera.data.lens = focal_length
        camera.data.dof.aperture_fstop = aperture

        ob = context['object']

        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                print("FOUND")
                ctx = bpy.context.copy()
                space = area.spaces[0]
                r = space.region_3d
                ctx['area'] = area
                ctx['region'] = area.regions[-1]
                ctx['space_data'] = space
                ctx['camera'] = camera
                r.view_location = ob.location
                r.view_distance = 2
                previous_lens = space.lens
                space.lens = camera.data.lens * zoomout_factor
                direction = Vector((view_point_x, view_point_y, view_point_z))
                r.view_rotation = lookat_viewport(r.view_location,
                                                  r.view_location + direction)
                bpy.ops.view3d.view_selected(ctx)
                bpy.ops.view3d.camera_to_view(ctx)
                space.lens = previous_lens
                camera.data.dof.focus_object = ob






Control = CameraControl
