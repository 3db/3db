Getting started with 3DB
========================

In this page, we'll go through all the steps to run a 3DB experiment
out-of-the-box.

Super-Quickstart
----------------

To get started with 3DB, clone the repository:

.. code-block:: bash
    
    git clone https://github.com/3db/3db

Setup 3DB by running this single command:

.. code-block:: bash
    
    curl https://raw.githubusercontent.com/3db/installers/main/linux_x86_64.sh | bash /dev/stdin threedb

Activate 3DB's conda env:

.. code-block:: bash

    conda activate threedb


Each 3DB experiment requires a ``BLENDER_DATA`` folder that contains two subfolders:

    * ``blender_models/``, which contains 3D models of objects (each 3D model is a ``.blend`` file with a single object)
    * ``blender_environments/``, which contains environments or backgrounds on which we want to render the objects.

Where will you get these 3D models and environments from? We got your back! We provide an `example repository <https://github.com/3db/blog_demo>`_ that has all what you need. First, clone this repository:

.. code-block:: bash

    git clone https://github.com/3db/blog_demo

Next, create a link to the 3D models and environments for one of the examples in ``blog_demo/data``. There are three available experiments there 

    * ``backgrounds``: render 3D models on various backgrounds.
    * ``texture_swap``: render a 3D model with various textures.
    * ``part_of_object``: render a 3D model in various poses and create an attribution heatmap. 
    
Here, we focus on the ``backgrounds`` experiment. Check out `this README <https://github.com/3db/blog_demo#running-this-demo>_ for steps on how to run any of these experiments.
 



.. code-block:: bash

    export BLENDER_DATA=data/backgrounds 
    export RESULTS_FOLDER=results


Now, ``${BLENDER_DATA}/blender_environments`` contains several environments and `${BLENDER_DATA}/blender_models` contains the 3D model of a mug.

Now that we have the `BLENDER_DATA` directory setup we can proceed to run 3DB. We first need to define the output folder of 3DB:

6. Run 

.. code-block:: bash

    export RESULTS_FOLDER=results_backgrounds


.. tabs::

    .. tab:: base.yaml

        .. code-block:: yaml

            inference:
            module: 'torchvision.models'
            label_map: '/path/to/3db/resources/imagenet_mapping.json'
            class: 'resnet18'
            output_shape: [1000]
            normalization:
                mean: [0.485, 0.456, 0.406]
                std: [0.229, 0.224, 0.225]
            resolution: [224, 224]
            args:
                pretrained: True
            evaluation:
            module: 'threedb.evaluators.classification'
            args:
                classmap_path: '/path/to/3db/resources/ycb_to_IN.json'
                topk: 1
            render_args:
            engine: 'threedb.rendering.render_blender'
            resolution: 256
            samples: 16
            policy:
            module: "threedb.policies.random_search"
            samples: 100
            logging:
            logger_modules:
                - "threedb.result_logging.image_logger"
                - "threedb.result_logging.json_logger"

    .. tab:: backgrounds.yaml

        .. code-block:: yaml

            base_config: "base.yaml"
            policy:
            module: "threedb.policies.random_search"
            samples: 20
            controls:
            - module: "threedb.controls.blender.orientation"
            - module: "threedb.controls.blender.camera"
                zoom_factor: [0.7, 1.3]
                aperture: 8.
                focal_length: 50.
            - module: "threedb.controls.blender.denoiser"

    .. tab:: texture_swaps.yaml

        .. code-block:: yaml

            base_config: "base.yaml"
            controls:
            - module: "threedb.controls.blender.orientation"
                rotation_x: -1.57
                rotation_y: 0.
                rotation_z: [-3.14, 3.14]
            - module: "threedb.controls.blender.position"
                offset_x: 0.
                offset_y: 0.5
                offset_z: 0.
            - module: "threedb.controls.blender.pin_to_ground"
                z_ground: 0.25
            - module: "threedb.controls.blender.camera"
                zoom_factor: [0.7, 1.3]
                view_point_x: 1.
                view_point_y: 1.
                view_point_z: [0., 1.]
                aperture: 8.
                focal_length: 50.
            - module: "threedb.controls.blender.material"
                replacement_material: ["cow.blend", "elephant.blend", "zebra.blend", "crocodile.blend", "keep_original"]
            - module: "threedb.controls.blender.denoiser"

    .. tab:: part_of_object.yaml

        .. code-block:: yaml

            base_config: "base.yaml"
            policy:
            module: "threedb.policies.random_search"
            samples: 20
            controls:
            - module: "threedb.controls.blender.orientation"
            - module: "threedb.controls.blender.camera"
                zoom_factor: [0.7, 1.3]
                aperture: 8.
                focal_length: 50.
            - module: "threedb.controls.blender.denoiser"


Next, let's run 3DB on a predefined config file, which you can find at `backgrounds.yaml`. This can be done by running the following two commands separately (e.g., in two separate terminal windows):

.. code-block:: python

    threedb_master $BLENDER_DATA backgrounds.yaml $RESULTS_FOLDER 5555
    threedb_workers 1 $BLENDER_DATA 5555

The first runs the master node which schedules the rendering tasks for the clients. This will keep running until all the rendering tasks are complete.
The second command runs the clients (here, it runs 1 client only), which performs the rendering.

A few seconds later, you will have your first results in `results_backgrounds/`! You can explore the results in a web interface by
running: 

.. code-block:: python

    python -m threedb.dashboard $RESULTS_FOLDER

This page will display the results as a large .json string.

To view the results using the full dashboard, simply paste the URL of the page displaying the .json string into the top of the page: https://3db.github.io/dashboard/.

In the sections below, we'll break down how to write your own config files,
and other ways to view your results. For advanced users, the `Extending 3DB <extending.html>`_
section of the documentation will help you customize and exploit the
modularity of 3DB.

=========

Writing a configuration file
----------------------------
There are six key parts of a 3DB configuration file. These are the
``evaluation``, ``rendering``, ``controls``, and
``logging``. Here, we'll go through each of these sections individually and
explain the required keywords, possible settings, and customization options for
each. 

Inference settings
""""""""""""""""""
The first step is to declare the inference model that will be evaluated by 3DB
by filling in a configuration under the ``inference`` keyword. The ``module``,
``class`` and ``args`` keywords tell 3DB how to instantiate the prediction
model. For example, for a pre-trained ResNet-50 classifier:

.. code-block:: yaml

    inference:
        module: 'torchvision.models.resnet'
        class: 'resnet50'
        args:
            pretrained: True

For a pre-trained object detection model:

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

Finally, the remaining arguments are for ``output_shape`` and ``class_mapping``.
The former tells 3DB how much space to allocate to save the model output; for
classifiers, this is just ``[NUM_CLASSES]``, whereas for detection models, we
will use ``[N, 6]`` where ``N`` is an upper bound on the number of bounding
boxes we will save for a given image (the 6 is because bounding boxes are
typically stored as ``(x1, y1, x2, y2, score, class)``. The ``class_mapping``
argument is optional and only used by some loggers---you can provide the path to
a JSON array containing class names, so that the output is more human-readable
(e.g., in the `dashboard`_ [TODO] ).

An example of a final inference configuration for an object detection experiment
is thus:

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
        label_map: './resources/coco_mapping.json'
        output_shape: [100, 6]

Evaluation settings
"""""""""""""""""""
The evaluator module is responsible for taking the output of the inference
model, and returning 

By default, 3DB provides default evaluators for both classification and object
detection models: different modalities/tasks (e.g., segmentation or regression)
will require implementation of a custom evaluator, which we outline in
the `Customizing 3DB <custom_evaluator.html>`_ section of the documentation.
