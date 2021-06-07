Using a Custom Infernce Model
===========================================

A key element of 3DB is the inference model that is being debugged. Although 3DB comes with default inference modules, users can use their own infernece models. Here we show you how to do that.

Out of the box, the 3DB supports:

* Classification (:mod:`torchvision.models`)
* Object Detection (:mod:`torchvision.models.detection`)

These modules can be used as we previously shown `here <writing_config_file.html#inference-settings>`__. For example, to use a pretrained ``ResNet-18`` torchvision model, the user shall add the following to the YAML configuration file:

.. code-block:: yaml

    inference:
        module: torchvision.models.resnet
        class: resnet50
        args:
            pretrained: True


Now in order to use your own inference (PyTorch) model, the first step is to create a pytorch module that defines your model. For example, create a folder with an empty file ``__init__.py``, and another file ``my_3db_inference_module.py`` that contains the following code:

.. code-block:: python 

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MyClassifier(torch.nn.modules):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


Then the only thing left to do is to add this new module
to your configuration file in the ``inference`` section:

.. code-block:: yaml

  logging:
    module: path.to.my.module
    class: MyClassifier
    args:
        arg1: "value for arg1"
        arg2: "value for arg2"

Here, ``path.to.my.module`` should point to the file containing your custom
inference class (i.e. ``my_3db_inference_module`` from the above example). 
In general, you can make your custom inference module available in 
any way you see fit, for instance:

* Make a pip package.
* Add the proper folder to ``$PYTHON_PATH``.
* Create and install a local package.

Note that if you add a module for solving a task different than image classification or object detection, you will need to also add a custom evaluator as we describe in `here <custom_evaluator.html>`__. Otherwise, you can use the 3DB builtin evaluators found in :mod:`threedb.evaluators`.