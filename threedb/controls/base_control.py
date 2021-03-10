"""Defines the BaseControl base class
"""

from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Any, Tuple


class BaseControl(ABC):
    """Abstract base class for 3DB controls

    The goal of controls two folds:

    - They declare a number of parameters, discrete or continuous.
    - Based on values for these parameters they influence the image that
      will be rendered

    There are two kinds of Controls:

    - `pre`: They update the scene before it is rendered
    - `post`: They update the rendered 2D image.

    Note
    ----
    The attributes can be defined in the class itself if they are constant.
    However, it is possible to define them in the __init__ function. This
    is especially useful for control that depend on data (like material swaps).

    """

    @property
    @abstractproperty
    def kind(self) -> str:
        """str: The kind of control

        It should be either 'pre' or 'post'
        """
        raise NotImplementedError

    @property
    def continuous_dims(self) -> Dict[str, Tuple[float, float]]:
        """ Describes the set of continuous parameters this control needs.

        It is of the shape: parameter_name -> (min_value, max_value)
        """
        return {}

    @property
    def discrete_dims(self) -> Dict[str, list[Any]]:
        """Describes the set of discrete parameters this control needs.

        It is of the shape: parameter_name -> [value_1, value_2, ..., value_n]
        """
        return {}

    def __init__(self, root_folder: str):
        """Construct a BaseControl

        Parameters
        ----------
        root_folder
            The folder containing all the data for this 3DB experiment. All
            paths are lative to his folder
        """
        self.root_folder = root_folder

    @abstractmethod
    def apply(self, target, **kwargs):
        """Modify the target give a combination of the inputs the control needs

        Note
        ----
        target will either be an object the control will modify to alter the
        scene if its `kind` is 'pre'. Otherwise it will be a 2D image in the
        shape of a pytorch tensor.

        Parameters
        ----------
        target
            The object to influnce, either context object or 2D image
        **kwargs
            a dict containing an entry for each of the discrete and continuous
            parameters declared by the Control

        Returns
        -------
        target
            For 'pre' controls, the target can be modified in place, for post
            a new image should be returned

        """
        raise NotImplementedError

    def unapply(self, context):
        """Undo the modification on a scene

        Note
        ----
        Most of the time, recreating a scene is very expensive, therefore,
        controls are asked to implement a reverse operation to undo their
        changes. Controls that need to store state in order to undo their
        actions should add data to the `target` object they received.

        Parameters
        ----------
        context
            The description of the scene to render
        """
