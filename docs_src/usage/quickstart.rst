Getting started with 3DB
========================

In this page, we'll go through all the steps to run a 3DB experiment
out-of-the-box.

Quickstart
----------------
Requirements
"""""""""""""

You will need a working Python 3.x installation. To follow the rest of the Quickstart, you will also need `Anaconda <https://docs.anaconda.com/anaconda/install/>`_.

Installation
"""""""""""""

To get started with 3DB, first clone the repository:

.. code-block:: bash
    
    git clone https://github.com/3db/3db

Next, run the following command to setup 3DB:

.. code-block:: bash
    
    curl https://raw.githubusercontent.com/3db/installers/main/linux_x86_64.sh | bash /dev/stdin threedb

Finally, activate 3DB's conda environment:

.. code-block:: bash

    conda activate threedb

You are now ready to start running 3DB experiments!

----

Run a simple experiment
"""""""""""""""""""""""
Now we will demonstrate how, in only few minutes, you can setup an experiment and generate the following images of a cup rendered with random orientations on various backgrounds.

.. thumbnail:: /_static/backgrounds_example.png
    :width: 700
    :group: background


Each 3DB experiment requires a ``BLENDER_DATA`` folder that contains two subfolders:

    * ``blender_models/``, containing 3D models of objects (each 3D model is a ``.blend`` file with a single object)
    * ``blender_environments/``, containing environments (backgrounds) on which we will render the objects

For this simple experiment, we provide an `example repository <https://github.com/3db/blog_demo>`_ that contains all the 3D models and environments you need.
Clone the example repository and navigate to that folder:

.. code-block:: bash

    git clone https://github.com/3db/blog_demo
    cd blog_demo

Then, update the ``BLENDER_DATA`` variable to point to the location of the 3D models and environments. In our case, set it as follows: 

.. code-block:: bash

    export BLENDER_DATA=data/backgrounds 

.. note::

    There are _`in-line targets` three available experiments in ``blog_demo``:
        * ``backgrounds``: renders a 3D models on various backgrounds.
        * ``texture_swap``: renders a 3D model with various textures.
        * ``part_of_object``: renders a 3D model in various poses, then creates an attribution heatmap.

    Here, we focus on the ``backgrounds`` experiment. Refer to `this README <https://github.com/3db/blog_demo#running-this-demo>`_ for steps on how to run the other experiments.
 
Next, define the output directory where 3DB will output the results.

.. code-block:: bash

    export RESULTS_FOLDER=results

The next step is to tell 3DB what configurations of 3D objects to render, how to evaluate the rendered images, and what data to log.
These should all be specified inside a ``YAML configuration file``. 

Here, we provide example YAML files, which are also in the same `example repository <https://github.com/3db/blog_demo>`_ that you already cloned.
Later on we will walk you through how to write your own configuration files.

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
                module: "threedb.controls.blender.orientation"
                module: "threedb.controls.blender.camera"
                    zoom_factor: [0.7, 1.3]
                    aperture: 8.
                    focal_length: 50.
                module: "threedb.controls.blender.denoiser"

    .. tab:: texture_swaps.yaml

        .. code-block:: yaml

            base_config: "base.yaml"
            controls:
                module: "threedb.controls.blender.orientation"
                    rotation_x: -1.57
                    rotation_y: 0.
                    rotation_z: [-3.14, 3.14]
                module: "threedb.controls.blender.position"
                    offset_x: 0.
                    offset_y: 0.5
                    offset_z: 0.
                module: "threedb.controls.blender.pin_to_ground"
                    z_ground: 0.25
                module: "threedb.controls.blender.camera"
                    zoom_factor: [0.7, 1.3]
                    view_point_x: 1.
                    view_point_y: 1.
                    view_point_z: [0., 1.]
                    aperture: 8.
                    focal_length: 50.
                module: "threedb.controls.blender.material"
                    replacement_material: ["cow.blend", "elephant.blend", "zebra.blend", "crocodile.blend", "keep_original"]
                module: "threedb.controls.blender.denoiser"

    .. tab:: part_of_object.yaml

        .. code-block:: yaml

            base_config: "./base.yaml"
            render_args:
                engine: 'threedb.rendering.render_blender'
                resolution: 256
                samples: 16
                with_uv: True
            controls:
                module: "threedb.controls.blender.orientation"
                    rotation_x: -1.57
                    rotation_y: 0.
                    rotation_z: [-3.14, 3.14]
                module: "threedb.controls.blender.camera"
                    zoom_factor: [0.7, 1.3]
                    view_point_x: 1.
                    view_point_y: 1.
                    view_point_z: 1.
                    aperture: 8.
                    focal_length: 50.
                module: "threedb.controls.blender.denoiser"
                module: "threedb.controls.blender.background"
                    H: 1.
                    S: 0.
                    V: 1.

The first file, ``base.yaml``, contains common configurations that are used by the three other YAML files.
Each of the other YAML files corresponds to one of the aformentioned experiments.
We will use the ``backgrounds.yaml`` already present in the example repository.

----

You are now ready to run 3DB! First, run the ``master node``, which schedules the rendering tasks (for clients). This will keep running until all the rendering tasks are complete:

.. code-block:: bash

    threedb_master $BLENDER_DATA backgrounds.yaml $RESULTS_FOLDER 5555

In a separate terminal window, run the ``client``, which performs the rendering.
To do so, first make sure that 3DB's conda environment is activated and that the ``BLENDER_DATA`` variable is properly set.

.. code-block:: bash

    conda activate threedb
    cd blog_demo
    export BLENDER_DATA=data/backgrounds

Then run 1 client (you can run multiple clients in parallel to speed up the rendering) using the following line of code:
 
.. code-block:: bash

    threedb_workers 1 $BLENDER_DATA 5555


A few seconds later, you will have your first results in ``results/``! You can explore the results in a web interface by running: 

.. code-block:: bash

    python -m threedb.dashboard $RESULTS_FOLDER

This page will display the results as a large .json string.

To view the results using the full dashboard, simply paste the URL of the page displaying the .json string into the top of this page: https://3db.github.io/dashboard/.
Below are examples of rendered images that you will see in the dashboard!

.. thumbnail:: /_static/dashboard_example.png
    :width: 700
    :group: background

You can also read the .json log file into ``pandas``, and analyze the results. For example, you can run the following python script, which is also in the demo repository: 

.. tabs::

    .. tab:: analyze_backgrounds.py

        .. code-block:: python

            import pandas as pd
            import numpy as np
            import json

            log_lines = open('results/details.log').readlines()
            class_map = json.load(open('results/class_maps.json'))
            df = pd.DataFrame.from_records(list(map(json.loads, log_lines)))
            df['prediction'] = df['prediction'].apply(lambda x: class_map[x[0]])
            df['is_correct'] = (df['is_correct'] == 'True')

            res = df.groupby('environment').agg(accuracy=('is_correct', 'mean'),
                    most_frequent_prediction=('prediction', lambda x: x.mode()))
            print(res)

*Congratulations! You have successfully completed your first 3DB experiment!*

In the sections below, we'll break down how to write your own configuration files,
and other ways to view your results. For advanced users, the `Extending 3DB <extending.html>`_
section of this documentation will help you customize and exploit the
modularity of 3DB.

=========

Writing a configuration file
----------------------------
There are six key parts of a 3DB configuration file:
    
    * ``inference``: defines some inference model to predict on the rendered images.
    * ``evaluation``: defines what evaluation metrics to compute given the output from the inference model.
    * ``rendering``: defines rendering-specific settings and arguments. 
    * ``controls``: defines the set of transformations to apply to the 3D model/environment before rendering the scene.
    * ``policy``: defines how to search through the various controls configurations.
    * ``logging``: defines how the results of 3DB are saved (e.g. JSON, Images, TensorBoard).

An example of each can be found in the YAML files of the above simple experiment. We will now go through each of these sections individually and
explain the required keywords, possible settings, and customization options for
each. 

Inference settings
""""""""""""""""""
The first step is to declare the inference model that will be evaluated by 3DB
by filling in a configuration under the ``inference`` keyword. The ``module``,
``class`` and ``args`` keywords tell 3DB how to instantiate the prediction
model. Below are examples showing how to instantiate a pre-trained ResNet-50 classifier and a pretrained object detection model, respectively:

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

Finally, the remaining arguments are for ``output_shape`` and ``label_map``.
The former tells 3DB how much space to allocate to save the model output:
    
    * for classifiers, this is just ``[NUM_CLASSES]``
    * for detection models, we will use ``[N, 6]`` where ``N`` is an upper bound on the number of bounding boxes we will save for a given image (the 6 is because bounding boxes are typically stored as ``(x1, y1, x2, y2, score, class)``. 
    
The ``label_map`` argument is optional and only used by some loggers---you can provide the path to a JSON array containing class names, so that the output is more human-readable.

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
model, and returning some evaluation metrics. 

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
                    classmap_path: './resources/uid_to_COCO.json'



Different modalities/tasks (e.g., segmentation or regression)
will require implementing custom evaluators, which we outline in
the `Customizing 3DB <custom_evaluator.html>`_ section of the documentation.


Rendering settings
"""""""""""""""""""
This part of the config file is responsible for declaring rendering-specific parameters and configurations, e.g., which renderer to choose, what image sizes to render, how many ray-tracing samples to use and so forth. The currently supported keywords for this section of the config file are:
    * `engine`: which renderer to use. 3DB supports Blender by default, :class:`threedb.rendering.render_blender.Blender`. See `Customizing 3DB <custom_renderer.html>`_ for how to add custom renderers.
    * `resolution`: the resolution of the rendered images.
    * `samples`: number of sample used for ray-tracing.

Here is an example of these settings:

.. code-block:: yaml

    render_args:
        engine: 'threedb.rendering.render_blender'
        resolution: 256
        samples: 16

which specify blender as the renderer, a resolution of 256x256 of the rendered images, and 16 as the number of samples for ray-tracing in Blender.  

Controls settings
"""""""""""""""""""
Every experiment requires the user to define how they want to reconstruct/manipulate the scene, e.g.

    * where will the object be placed
    * what is the orientation of the object
    * what is the background of the object
    * is there anything occluding the object

and the list goes on. In order to do these and control the scene, a list of ``controls`` has to be defined in the YAML file.

Policy settings
"""""""""""""""""""

Logging settings
"""""""""""""""""""
