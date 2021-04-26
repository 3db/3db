Customizing Rendering
=====================

This customization is intended for advanced users familiar with 3D rendering. 3DB comes with `blender` as the default renderer, 
but users can choose their preferred renderer instead. 

The first step is to subclass the provided abstract base class,
:class:`threedb.rendering.base_renderer.BaseRenderer`:

The user then has to implement these functions: ``__init__``, ``enumerate_models``, ``enumerate_environments``, 
``declare_outputs``, ``load_model``, ``get_model_uid``,  ``load_env``, ``setup_render``,  ``get_context_dict``, 
and ``render``. 

Please check :class:`threedb.rendering.base_renderer.BaseRenderer` for the functionality of each
of these functions.

We also encourage advanced users to check the blender render class :class:`threedb.rendering.render_blender.Blender` for example
implementations of the above functions.
