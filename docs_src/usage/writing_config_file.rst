Writing a configuration file
----------------------------
There are six key parts/modules of a 3DB configuration file:
    
    * ``inference``: defines the model that is used to make predictions on the rendered images.
    * ``evaluation``: defines what evaluation metrics to compute from the output from the inference model.
    * ``rendering``: defines rendering-specific settings and arguments. 
    * ``controls``: defines the set of transformations to apply to the 3D model/environment before rendering the scene.
    * ``policy``: defines how to search through the various controls configurations.
    * ``logging``: defines how the results of 3DB are saved (e.g. JSON, Images, TensorBoard).

An example of each can be found in the YAML files of the above simple experiment. We will now go through each of these sections individually and
explain the required keywords, possible settings, and customization options for
each. 

Inference settings
""""""""""""""""""
The first step is to declare the inference model that 3DB will use to make predictions
by filling in a configuration under the ``inference`` keyword. The ``module``,
``class``, and ``args`` keywords tell 3DB how to instantiate the prediction
model. Below are examples showing how to instantiate a pre-trained ResNet-50 classifier and a pre-trained object detection model, respectively:

.. tabs::

    .. tab:: Pre-trained ResNet-50 Classifier

        .. code-block:: yaml

            inference:
                module: 'torchvision.models.resnet'
                class: 'resnet50'
                args:
                    pretrained: True

    .. tab:: Pre-trained Object Detector

        .. code-block:: yaml
        
            inference:
                module: 'torchvision.models.detection'
                class: 'retinanet_resnet50_fpn'
                args:
                    pretrained: True


Next, we provide the ``normalization`` and ``resolution`` arguments, both of
which are used to pre-process inputs before they are fed to the inference model:

.. code-block:: yaml

    inference:
        module: 'torchvision.models.detection'
        class: 'retinanet_resnet50_fpn'
        args:
            pretrained: True
        ## --- NEW STUFF ---
        normalization:
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
        resolution: [500, 500]
        ## --- /END NEW STUFF ---

Finally, the remaining argument to specify is ``label_map``. This argument is optional and only used by some loggers---you can provide the path to a JSON array containing class names, so that the output is more human-readable.

An example of a complete inference configuration for an object detection experiment is the following:

.. code-block:: yaml

    inference:
        module: 'torchvision.models.detection'
        class: 'retinanet_resnet50_fpn'
        args:
            pretrained: True
        normalization:
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
        resolution: [500, 500]
        label_map: '/path/to/3db/resources/coco_mapping.json'


The user might want to use a custom inference model that goes beyond what 3DB comes with. We outline how to do that in
the `Customizing 3DB <custom_inference.html>`__ section of the documentation.

Evaluation settings
"""""""""""""""""""
The evaluator module is responsible for taking the output of the inference
module and returning evaluation metrics. 

By default, 3DB provides evaluators for both classification and object
detection models: 


.. tabs::

    .. tab:: Image Classification

        .. code-block:: yaml

            evaluation:
                module: 'threedb.evaluators.classification'
                args:
                    classmap_path: '/path/to/3db/resources/ycb_to_IN.json'
                    topk: 1

    .. tab:: Object Detection

        .. code-block:: yaml
        
            evaluation:
                module: "threedb.evaluators.detection"
                args:
                    iou_threshold: 0.5
                    classmap_path: '/path/to/3db/resources/uid_to_COCO.json'



Different modalities/tasks (e.g., segmentation or regression)
will require implementing custom evaluators, which we outline in
the `Customizing 3DB <custom_evaluator.html>`__ section of the documentation.


Rendering settings
"""""""""""""""""""
This part of the config file is responsible for declaring rendering-specific parameters and configurations, e.g., which renderer to choose, what image sizes to render, how many ray-tracing samples to use and so forth. The currently supported keywords for this section of the config file are:

    * ``engine``: which renderer to use. 3DB supports Blender by default, :class:`threedb.rendering.render_blender.Blender`. See `Customizing 3DB <custom_renderer.html>`__ for how to add custom renderers.
    * ``resolution``: the resolution of the rendered images.
    * ``samples``: number of sample used for ray-tracing.
    * ``with_segmentation``: if ``True``, returns a segmentation map along with an RGB image. Defaults to ``False``.
    * ``with_depth``: if ``True``, returns a depth map along with an RGB image. Defaults to ``False``.
    * ``with_uv``: if ``True``, returns a UV map along with an RGB image. Defaults to ``False``.


Here is an example of these settings, where only RGB and segmentation images are returned by 3DB:

.. code-block:: yaml

    render_args:
        engine: 'threedb.rendering.render_blender'
        resolution: 256
        samples: 16
        with_segmentation: True
        with_depth: False
        with_uv: False

Controls settings
"""""""""""""""""""
Every experiment requires the user to define how they want to control/manipulate the scene, e.g.,

    * where will the object be placed?
    * what is the orientation of the object?
    * what is the background of the object?
    * is there anything occluding the object?

In order to control/manipulate the scene, a list of ``controls`` has to be defined in the YAML file. A number of example controls are shown below.

.. tabs::

    .. tab:: Orientation

        .. code-block:: yaml

            controls:
                module: threedb.controls.blender.orientation
                    rotation_x: [-3.14, 3.14]
                    rotation_y: [-3.14, 3.14]
                    rotation_z: [-3.14, 3.14]

    .. tab:: Background

        .. code-block:: yaml
        
            controls:
                module: threedb.controls.blender.background
                    H: [0.0, 1.0]
                    S: [0.0, 1.0]
                    V: [0.0, 1.0]

    .. tab:: Denoiser

        .. code-block:: yaml
        
            controls:
                module: threedb.controls.blender.denoiser


    .. tab:: Position

        .. code-block:: yaml
        
            controls:
                module: threedb.controls.blender.position
                    offset_x: [-0.02, 0.02]
                    offset_y: [-0.02, 0.02]
                    offset_z: [-0.02, 0.02]

3DB comes with a set of predefined controls that the user can use. These can be found in :mod:`threedb.controls`. The user can also add custom controls if desired, see `Customizing 3DB <custom_controls.html>`__ for how to add new controls.

Policy settings
"""""""""""""""""""
After specifying the controls to apply to specific objects/scenes, the user must specify how they want to search over the control space.
This should be done in the configuration file under policy settings.
We provide a number of default search policies that the user can directly use in :mod:`threedb.policies`. 

For example, a user might want to randomly search in the space of poses of objects, or do a grid search over specific object poses. We provide example configuration files for each case in the code block below:


.. tabs::

    .. tab:: Random Search

        .. code-block:: yaml

            base_config: base.yaml
            controls:
                module: threedb.controls.blender.camera
                    zoom_factor: 1.
                    aperture: 8.
                    focal_length: 50.
                module: threedb.controls.blender.orientation
                    rotation_x: [-3.14, 3.14]
                    rotation_y: [-3.14, 3.14]
                    rotation_z: [-3.14, 3.14]
            policy:
                module: "threedb.policies.random_search"
                samples: 5

    .. tab:: Grid Search

        .. code-block:: yaml
        
            base_config: base.yaml
            controls:
                module: threedb.controls.blender.camera
                    zoom_factor: 1.
                    aperture: 8.
                    focal_length: 50.
                module: threedb.controls.blender.orientation
                    rotation_x: [-3.14, 3.14]
                    rotation_y: [-3.14, 3.14]
                    rotation_z: [-3.14, 3.14]
            policy:
                module: "threedb.policies.grid_search"
                samples: 5

The currently supported keywords for ``policy`` in the config file are:

    + ``module``: which policy to use from :mod:`threedb.policies`.
    + ``samples``: number of samples to search according to a given policy. For random search, this will be the number of random samples. For grid search, this will be the number of vertices on the grid.


Logging settings
"""""""""""""""""""
Finally, the user has to specify how to log or dump the result logs generated by 3DB.
The output returned by each 3DB rendering consists of the rendered image(s), the prediction (based on the evaluation module), the control parameters of the current render, in addition to several other pieces of meta-data (object ID, image ID, etc).
Parts of this information can be dumped into JSON files, parts can be saved as image files, and other parts can be saved via other loggers as well.

3DB thus comes with a number of default ``loggers`` that allow the user to easily read the data. These can be found in :mod:`threedb.result_logging`. Here are snippets of how to add each logger type to your YAML file.

.. tabs::

    .. tab:: Image Logger

        .. code-block:: yaml

            logging:
                logger_modules: 
                    threedb.result_logging.image_logger

    .. tab:: JSON Logger

        .. code-block:: yaml
        
            logging:
                logger_modules: 
                    threedb.result_logging.json_logger

    .. tab:: TensorBoard Logger

        .. code-block:: yaml
        
            logging:
                logger_modules: 
                    threedb.result_logging.tb_logger

    .. tab:: Dashboard Loggers

        .. code-block:: yaml
        
            logging:
                logger_modules: 
                    threedb.result_logging.image_logger
                    threedb.result_logging.json_logger


The user can also use any of these loggers simultaneously by adding them under each other (as done in ``Dashboard Loggers``).
For adding custom loggers, see `Customizing 3DB <custom_logger.html>`__.