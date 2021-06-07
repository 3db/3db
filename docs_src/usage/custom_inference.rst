Using a Custom Inference Model
===========================================

A key element of 3DB is the inference model that is being debugged.
That is, what model do you want to use to make inferences on the images rendered by 3DB?
3DB comes with two default inference modules, which use pre-trained models from torchvision.
Here, we further show how you can use your own custom inference models.

Out of the box, 3DB supports pre-trained models for:

* Classification (:mod:`torchvision.models`)
* Object Detection (:mod:`torchvision.models.detection`)

We `previously discussed <quickstart.html#inference-settings>`__ how to use the default modules with pre-trained models.
For example, to use a pre-trained ``ResNet-18`` model from torchvision, the user can simply add the following to their YAML configuration file:

.. code-block:: yaml

    inference:
        module: torchvision.models.resnet
        class: resnet50
        args:
            pretrained: True


Now in order to use your own (PyTorch) inference model, the first step is to create a PyTorch module that defines your model.
For example, first create a folder ``myinference``. Inside the folder, create an empty file ``__init__.py``, and another file ``my_3db_inference_module.py`` that contains the following code:

.. code-block:: python 

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MyClassifier(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
                nn.MaxPool2d(2), #kernel_size
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 192, 3, padding = 1),
                nn.MaxPool2d(2),
                nn.ReLU(inplace = True),
                nn.Conv2d(192, 384, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(384, 256, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, 3, padding = 1),
                nn.MaxPool2d(2),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, 3, padding = 1),
                nn.MaxPool2d(2),
                nn.ReLU(inplace = True)
            )
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 7 * 7, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Linear(4096, output_dim),
            )

        def forward(self, x):
            x = self.features(x)
            h = x.view(x.shape[0], -1)
            x = self.classifier(h)
            x = x.squeeze(0)
            return x, h


Next, simply point to the location of this new module in the ``inference`` section of your YAML file:

.. code-block:: yaml

  inference:
    module: path.to.my.module
    class: MyClassifier
    args:
        output_dim: 1000
    # You will need to re-define the following parameters even if they are in the base.yaml file that you import from
    normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    output_shape: [1000]
    resolution: [224, 224]

Here, ``path.to.my.module`` should point to the file containing your custom
inference class (i.e., ``my_3db_inference_module`` in the above example). 
In general, you can make your custom inference module available in 
any way you see fit, for instance:

* Make a pip package
* Add the proper folder to ``$PYTHON_PATH``
* Create and install a local package

In this particular example, the model we load has randomly initialized weights.
To load a model with pre-trained weights, you can modify the module's ``__init__`` function to load those weights (e.g. by passing in the path to a checkpoint as a parameter to ``__init__``).

Finally, note that if you add a module for solving a task other than image classification and object detection, you will also need to add a custom evaluator, which we describe `here <custom_evaluator.html>`__.
Otherwise, you can use the 3DB's built-in evaluators found in :mod:`threedb.evaluators`.