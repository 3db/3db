"""Abstract base classes for 3DB controls

The goal of control is two-fold:

- They declare a number of parameters, discrete or continuous.
- Based on values for these parameters they influence the image that will be
  rendered.

There are two kinds of Controls:

- *Pre-processing controls* update the scene before it is rendered (e.g., object
  orientation changes)
- *Post-processing controls* update the rendered 2D image (e.g., increasing
  contrast or adding noise)

This module file contains abstract base classes for both control types---new
controls can be introduced by sublcassing the appropriate one (see [TODO] for
details).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

import torch as ch

class BaseControl(ABC):
    """An Abstract Base Class superclassing both pre- and post-processing
    controls. Implements the required cross-control properties
    ``continuous_dims`` and ``discrete_dims`` as well as a standard initializer.
    """
    @property
    def continuous_dims(self) -> Dict[str, Tuple[float, float]]:
        """Describes the set of continuous parameters this control needs.

        It is of the shape: parameter_name -> (min_value, max_value)
        """
        return {}

    @property
    def discrete_dims(self) -> Dict[str, List[Any]]:
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

    def check_arguments(self, control_args: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks a dictionary of control arguments against the arguments
        pre-declared by ``self.continuous_dims`` and ``self.discrete_dims``

        Parameters
        ----------
        control_args : Dict[str, Any]
            A dictionary of control arguments.

        Returns
        -------
        Tuple[bool, str]
            A tuple indicating whether the control arguments are valid. If they
            are, the first return value is ``True`` and the second should be
            ignored. If not, the first return value is ``False`` and the second
            is an error message.
        """
        all_keys = self.continuous_dims.keys() | self.discrete_dims.keys()
        if control_args.keys() != all_keys:
            return False, 'Keys in control arguments do not match declared keys'
        if not all(isinstance(control_args[k], float) for k in self.continuous_dims):
            return False, 'Some continuous arguments are not of type float'
        if not all(control_args[k] in v for (k, v) in self.discrete_dims.items()):
            return False, 'Some discrete arguments do not match a declared valid value'
        return True, ''

class PostProcessControl(BaseControl, ABC):
    """An abstract sublass of ``BaseControl`` for post-processing controls.
    Differs from pre-processing controls in that the type signature of
    ``apply``.
    """
    @abstractmethod
    def apply(self, render: ch.Tensor, control_args: Dict[str, Any]) -> ch.Tensor:
        """Modify a rendered image and return the transformed output.

        Parameters
        ----------
        render : ch.Tensor
            A tensor representation of the rendered image.
        control_args: Dict[str, Any]
            Control-specific settings (e.g., noise level for noise corruption,
            contrast level for contrast change, etc.).

        Returns
        -------
        ch.Tensor
            The post-processed output.
        """
        raise NotImplementedError

class PreProcessControl(BaseControl, ABC):
    """An abstract sublass of ``BaseControl`` for post-processing controls.
    Differs from pre-processing controls in that the type signature of
    ``apply``, and the presence of an ``unapply`` function undoing the effects
    of the control.
    """

    @abstractmethod
    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Modify the target give a combination of the inputs the control needs

        Parameters
        ----------
        context: Dict[str, Any]
            A dictionary representation of the current scene.
        control_args: Dict[str, Any]
            a dict containing an entry for each of the discrete and continuous
            parameters declared by the Control

        Modifies the scene in-place, no return value.
        """
        raise NotImplementedError

    @abstractmethod
    def unapply(self, context: Dict[str, Any]) -> None:
        """Undo the modification on a scene

        Note
        ----
        Most of the time, recreating a scene is very expensive, therefore,
        controls are asked to implement a reverse operation to undo their
        changes. Controls that need to store state in order to undo their
        actions should add data to the `target` object they received.

        Parameters
        ----------
        context : Dict[str, Any]
            The description of the scene to render
        """
        raise NotImplementedError
