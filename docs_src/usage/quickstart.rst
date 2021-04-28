Getting started with 3DB
========================

In this page, we'll go through all the steps to run a 3DB experiment
out-of-the-box.

Super-Quickstart
----------------

To get started with sandbox, right away, 
```
0. Clone `https://github.com/3db/3db` and `https://github.com/3db/blog_demo`
1. Setup 3DB by running `curl https://raw.githubusercontent.com/3db/installers/main/linux_x86_64.sh | bash /dev/stdin threedb`
2. Activate 3db's conda env: `conda activate threedb`
```


Each 3DB experiments requires a `BLENDER_DATA` folder that contains two subfolders: 
```
 - `blender_models/` that containing 3D models (`.blend` files with a single object whose name matches the filename)
 - `blender_environments/` containing environments. We will provide you with these later.
```
Here is an example demo that has this folder already setup
```
3. Clone `https://github.com/3db/blog_demo`
4. Set `BLENDER_DATA=blog_demo/data/backgrounds`. 
```
`${BLENDER_DATA}/blender_environments` contains several backgrounds and `${BLENDER_DATA}/blender_models` contain the 3D model of a mug.

Now that we have the `BLENDER_DATA` directory setup we can proceed to run 3DB. We first need to define the output folder of 3DB:
```
5. Run `export RESULTS_FOLDER='results_backgrounds'`
```

Next, let's run 3DB on a predefined config file which you can find at `blog_demo/backgrounds.yaml`. This can be documentation
by running the following two commands:
```
    threedb_master $BLENDER_DATA backgrounds.yaml $RESULTS_FOLDER 5555
    threedb_workers 1 $BLENDER_DATA 5555
```
The first runs the master node which schudles the rendering tasks for the clients which are run by the second command (here 1 client only).

A few seconds later, you will have your first results in `results_backgrounds/`! You can explore them in a web interface by
running: 

```
python -m threedb.dashboard $RESULTS_FOLDER
```


In the sections below, we'll break down how to write your own config files,
and other ways to view your results. For advanced users, the `Extending 3DB <extending.html>`_
section of the documentation will help you customize and exploit the
modularity of 3DB.


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
