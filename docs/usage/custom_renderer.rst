Customizing Rendering
=====================

The basic 3DB functionality uses `blender` as the default renderer.
However, other user-specified renderers can easily be integrated
into 3DB as well. 

In particular, to add a custom renderer to 3DB, the first step is 
to subclass the provided abstract base class,
:class:`threedb.rendering.base_renderer.BaseRenderer`:

Then, one needs to implement the following functions into the custom
renderer class: ``__init__``, ``enumerate_models``, ``enumerate_environments``, 
``declare_outputs``, ``load_model``, ``get_model_uid``,  ``load_env``, ``setup_render``,  ``get_context_dict``, 
and ``render``. 

A detailed description of the functionality of each of these functions is provided in
:class:`threedb.rendering.base_renderer.BaseRenderer`. Further, example implementations
of these functions (for Blender) can be found in 
:class:`threedb.rendering.render_blender.Blender`.
