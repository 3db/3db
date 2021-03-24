Customizing Rendering
=====================

3DB comes with `blender` is the default renderer, but users can add there favorite renderer, e.g. `Mitsuba`. 
We will now go through the steps for implementing adding a new renderer to 3DB.

The first step to add a renderer is to subclass the provided base classes,
:class:`threedb.rendering.base_renderer.BaseRenderer`:

Then, the user has to implement these functions ``__init__``, ``enumerate_models``, ``enumerate_environments``, 
``declare_outputs``, ``load_model``, ``get_model_uid``,  ``load_env``, ``setup_render``,  ``get_context_dict``, 
and ``render``. 

Please check :class:`threedb.rendering.base_renderer.BaseRenderer` for the funcionality of each
of these functions. 

We also urge the user to check the blender render class :class:`threedb.rendering.render_blender.Blender` for example
implementation of the above functinos too.

