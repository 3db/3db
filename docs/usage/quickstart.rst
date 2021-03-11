Getting started with 3DB
========================

In this page, we'll go through all the steps to run a 3DB experiment
out-of-the-box.

.. image:: /_static/client_side_diagram.png


Data Collection
---------------


Writing a configuration file
----------------------------
There are four key parts of a 3DB configuration file. These are the
``inference``, ``evaluation``, ``rendering``, ``search``, ``controls``, and
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
will require implementation of a [TODO] custom evaluator, which we outline in
the `Customizing 3DB <TODO>`_ section of the documentation.