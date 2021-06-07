Using a Custom Evaluator (for custom tasks)
===========================================

Even though 3DB focuses on computer vision, it can also be used to solve a myriad of possible tasks.
We enable users to describe their own evaluation tasks in 3DB.

Out of the box, the frameworks supports:

* Classification (:mod:`threedb.evaluators.classification`)
* Object Detection (:mod:`threedb.evaluators.detection`)

The procedure for extending evaluators in 3DB is as follows:

#. Implement a subclass of the base class. For evaluators, it is: :mod:`threedb.evaluators.base_evaluator`.
#. Have your file export the class under the name expected by the framework. For evaluators, it is: ``Evaluator``.
#. Make sure it can be imported in your current environment (via ``$PYTHONPATH``, ``pip install -e ...``, ...).
#. Update the configuration file to use the newly defined evaluator.


Implementation
-----------

Each method is described in :mod:`threedb.evaluators.base_evaluator`. Your final file should look like this:

.. code-block:: python

    from typing import Dict, List, Tuple, Any
    from threedb.evaluators.base_evaluator import BaseEvaluator

    class ExampleEvaluator(BaseEvaluator):

        # The output_type variable needs to be defined; it can be any string.
        output_type = 'YourOutputType'
        # The KEYS variable need to match the keys in declare_outputs
        # and summary_stats.
        KEYS = ['is_correct', 'loss']

        def __init__(self, arg1, arg2):
            # In order to implement methods in this class you probably
            # will need some metadata. Feel free to accept any argument
            # you need in your constructor. You will be able to populate
            # them in your config file.

            self.arg1 = arg1
            self.arg2 = arg2

        def get_segmentation_label(self, model_uid: str) -> int:
            # The output of this function will be the value associated
            # to pixels that belongs to the object of interest.
            # If you don't plan on using segmentation you can output anything.
            return 0 


        def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
            # The goal of this method is to declare what kinds of metrics
            # the evaluator will generate.
            return {
                'is_correct': ([], 'bool'),
                'loss': ([], 'float32')
                # You can add as many metrics as you want
            }

        def get_target(self, model_uid: str, render_output: Dict[str, Tensor]) -> LabelType:
            # returns the expected label (whatever label means for this specific task)
            pass

        def to_tensor(self, pred: Any, *_) -> Tensor:
            # In 3DB we only work with Tensors, so this method is used to convert
            # the model output into a tensor.
            return pred

        def summary_stats(self, pred: ch.Tensor, label: LabelType) -> Dict[str, Output]:
            # This method is used to generate the metrics declared in
            # declare_outputs() using the output of to_tensor() and get_target().
            return {
                'is_correct': False,
                'loss': 0
                # You can add as many metrics as you want
            }

    # IMPORTANT! Needed so that threedb is able to import your custom evaluator
    # (since it can't know how you named your class).
    Evaluator = ExampleEvaluator

Updating the configuration file
-------------------------------

You should update the ``evaluation`` section of your configuration file:

.. code-block:: yaml

    evaluation:
        module: "path.to.your.newly.created.module"
        args:
            arg1: "value for arg1"
            arg2: "value for arg2"
