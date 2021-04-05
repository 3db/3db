Using a Custom Evaluator (for custom tasks)
===========================================

Even if ``threedb`` focuses on computer vision, there might be myriad of possible tasks one might want to solve.
Out of the box, the frameworks supports:

* Classification (:mod:`threedb.evaluators.classification`)
* Object Detection (:mod:`threedb.evaluators.detection`)

However, we want users to be able to describe their task, whatever it is, and be able to debug/evaluate their
models with ``threedb``.

As for all possible ``threedb`` extensions, the procedure is the same for evaluators:

#. Implement a subclass of the base class for that feature, here: :mod:`threedb.evaluators.base_evaluator`.
#. Have your file export the class under the name expected by the framework, here: ``Evaluator``.
#. Make sure it can be imported in your current environment (``$PYTHONPATH``, ``pip install -e ...``, ...).
#. Update the configuration file to use the newly defined evaluator.


Subclassing
-----------

Since each method is described :mod:`threedb.evaluators.base_evaluator`, we won't repeat ourselves here. However, your file should looks like this:

.. code-block:: python

    from threedb.evaluators.evaluator import BaseEvaluator

    class ImageSegmentation(BaseEvaluator):

        def __init__(self, arg1, arg2):
            # In order to implement methods in this class you probably
            # will need some metadata, feel free to accept any argument
            # you need in your constructor. You will be able to populate
            # them in your config file

            self.arg1 = arg1
            self.arg2 = arg2

        def get_segmentation_label(model_uid: str) -> int:
            # The output of this function will be the value associated
            # to pixels that belongs to the object of interest
            # If you don't plan on using segmentation you can output anything
            return 0 


        def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
            # The goal of this method is to declare what kind of metrics
            # the evaluator will generate
            return {
                'is_correct': ([], 'bool'),
                'loss': ([], 'float32'),
                'F1': ([], 'float32'),
                'AUC': ([], 'float32')
                # You can add as many metrics as you want
            }

        def get_target(self, model_uid: str, render_output: Dict[str, Tensor]) -> LabelType:
            # returns the expected label (whatever is Label for this specific task)
            pass

        def to_tensor(self, pred: Any, *_) -> Tensor:
            # In 3DB we only work with Tensors, the goal of this method is to convert
            # whatever the output of the model is into a tensor
            return pred

        def summary_stats(self, pred: ch.Tensor, label: LabelType) -> Dict[str, Output]:
            # The goal of this method is to generate the metrics declared in
            # declare_outputs() using the output of to_tensor() and get_target()
            pass

    # IMPORTANT! Needed so that threedb is able to import your custom evaluator
    # (Since it can't know how you named your class)
    Evaluator = ImageSegmentation

Updating the configuration file
-------------------------------

You should update the ``evaluation`` section of your configuration file:

.. code-block:: yaml

    evaluation:
        module: "path.to.your.newly.created.module"
        args:
            arg1: "value for arg1"
            arg2: "value for arg2"