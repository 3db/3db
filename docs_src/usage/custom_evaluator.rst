Using a Custom Evaluator (for custom tasks)
===========================================

Even though 3DB focuses on computer vision, it can also be used to solve a myriad of possible tasks.
We enable users to describe their own evaluation tasks in 3DB.

Out of the box, the frameworks supports:

* Classification (:mod:`threedb.evaluators.classification`)
* Object Detection (:mod:`threedb.evaluators.detection`)

The procedure for extending evaluators in 3DB is as follows:

#. Implement a subclass of the base class :mod:`threedb.evaluators.base_evaluator`.
#. Have your file export the class under the name ``Evaluator``.
#. Make sure it can be imported in your current environment (via ``$PYTHONPATH``, ``pip install -e ...``, ...).
#. Update the configuration file to use the newly defined evaluator.

Implementation
---------------

Below, we provide an example implementation of custom evaluator for an image
segmentation task. We'll briefly outline each of the required steps below---for
more detailed documentation of each abstract function, see the
:mod:`threedb.evaluators.base_evaluator` module.

1. We'll start by importing the required modules and subclassing the
   :mod:`~threedb.evaluators.base_evaluator.BaseEvaluator` class. In order for
   the subclass to be valid, we need to declare two variables, ``KEYS`` and
   ``output_type``. At a high level, the purpose of any evaluator is to map from
   a (model prediction, label) pair to an arbitrary dictionary summarizing the
   results---the ``KEYS`` variable declares what the keys of this dictionary
   will be. The ``output_type`` variable is only used by the dashboard to
   visualize the results, and can be any string.

    .. code-block:: python

        from typing import Dict, List, Tuple, Any
        import torch as ch
        import json
        from threedb.evaluators.base_evaluator import BaseEvaluator

        class ImageSegmentation(BaseEvaluator):
            # The output_type variable needs to be defined; it can be any string and
            # is only used by the dashboard to decide how the results are displayed.
            output_type = 'segmentation'

            # The KEYS variable needs to match the keys in declare_outputs
            # and summary_stats.
            KEYS = ['is_correct', 'loss', 'seg_map']
    
    
2. Next, we implement the init function: this can take arbitrary arguments and
   should set up everything that the evaluator needs to generate metadata about
   model predictions. For our segmentation evaluator, we'll need (a) a file
   mapping model UIDs to classes; (b) an error threshold at which to call a
   segmentation "correct;" and (c) the size of the segmentation maps produced.

    .. code-block:: python

        def __init__(self, model_uid_to_class_file, l2_thresh, im_size):
            # In order to implement methods in this class you probably
            # will need some metadata. Feel free to accept any argument
            # you need in your constructor. You will be able to populate
            # them in your config file.

            self.model_to_class = json.load(open(model_uid_to_class_file))
            self.l2_thresh = l2_thresh
            self.im_size = im_size

3. The next abstract function we have to implement is
   :meth:`~threedb.evaluators.base_evaluator.BaseEvaluator.get_segmentation_label`,
   which takes in a model uid and should return a label to be used in the
   segmentation map (i.e., the segmentation map will be -1 wherever the
   backgroiunds visible, and X wherever the model is visible, where X is the
   output of this function). Below is a function that uses the provided JSON
   file to return this label:

    .. code-block:: python

        def get_segmentation_label(self, model_uid: str) -> int:
            # The output of this function will be the value associated
            # to pixels that belongs to the object of interest.

            label = self.uid_to_targets[model_uid]
            return label[0] if isinstance(label, list) else label

4. Next, we implement the 
   :meth:`~threedb.evaluators.base_evaluator.BaseEvaluator.declare_outputs`
   function---this must return a dictionary with keys equal to the ``KEYS``
   variable declared earlier, and values equal to tuples of the form ``(shape,
   type)``. In particular, ``shape`` should be a list describing the shape of
   the tensor that will be returned, and ``type`` should be a PyTorch dtype:

    .. code-block:: python

        def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
            # The goal of this method is to declare what kinds of metrics
            # the evaluator will generate.
            return {
                'is_correct': ([], 'bool'),
                'loss': ([], 'float32'),
                'seg_map': (self.im_size, 'float32')
                # Any other metrics you want to report!
            }

5. The next step is to declare the ``get_target`` function, which takes in (a)
   the UID of the rendered 3D model and (b) the ``render_output`` dictionary
   consisting of the render output (for the built-in Blender engine, this
   thankfully already comes with a "segmentation" key!), and returns the
   desired ground-truth that our segmentation model's output will be compared
   to:

    .. code-block:: python

        def get_target(self, model_uid: str, render_output: Dict[str, Tensor]) -> LabelType:
            # returns the expected label (whatever label means for this specific task)
            return render_output['segmentation']
    
6. The last and most important step is the ``summary_stats`` function, which
   takes in a model prediction and a label (the output of ``get_target``) and
   returns a dictionary that has the same keys as KEYS, and values that are
   PyTorch tensors with the declared shapes and types from ``declare_outputs``.
   For example, assuming the model outputs a segmentation map, we might return
   something like the following:

    .. code-block:: python

        def summary_stats(self, pred: ch.Tensor, label: LabelType, input_shape: List[int]) -> Dict[str, Output]:
            # This method is used to generate the metrics declared in
            # declare_outputs() using the output of to_tensor() and
            # get_target().
            loss = (pred - label).norm()
            is_correct = ch.tensor(loss < self.l2_thresh)
            return {
                'is_correct': is_correct,
                'loss': loss,
                'seg_map': pred 
                # You can add as many metrics as you want as long 
                # as you match declare_outputs
            }

7. Finally, don't forget to export your class under the ``Evaluator`` variable!

    .. code-block:: python

        # IMPORTANT! Needed so that threedb is able to import your custom evaluator
        # (since it can't know how you named your class).
        Evaluator = ImageSegmentation

That's it! We've implemented all the functions needed to add a custom task to 3DB.

Updating the configuration file
-------------------------------

You should update the ``evaluation`` section of your configuration file:

.. code-block:: yaml

    # ... rest of YAML file
    evaluation:
        module: "path.to.your.newly.created.module"
        args:
            model_uid_to_class_file: "/path/to/mapping/file"
            l2_thresh: 10.
            im_size: [1, 224, 224]
    render_args:
        engine: 'threedb.rendering.render_blender'
        resolution: 256
        samples: 16
        with_segmentation: true
