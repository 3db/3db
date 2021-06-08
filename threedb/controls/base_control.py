"""
threedb.controls.base_control
==============================

Abstract base classes for 3DB controls

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
controls can be introduced by sublcassing the appropriate one (see `here <custom_controls.html>`_ for
details).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union

import torch as ch

class BaseControl(ABC):
    """An Abstract Base Class superclassing both pre- and post-processing
    controls. Implements the required cross-control properties
    ``continuous_dims`` and ``discrete_dims`` as well as a standard initializer.
    """

    @property
    def continuous_dims(self) -> Dict[str, Tuple[float, float]]:
        """Describes the set of continuous parameters this control needs.

        Returns
        -------
        Dict[str, Tuple[float, float]]
            Will be of the form: parameter_name -> (min_value, max_value)
        """
        return self._continuous_dims

    def update_continuous_dim(self, key: str, val: Tuple[float, float]):
        """Updates a specified continuous dimensions of the control with a
        user-provided value. Raises a ``ValueError`` if the key does not matched
        a pre-declared continuous control dimension.

        Parameters
        ----------
        key : str
            The key of the search dimension to override.
        val : Tuple[float, float]
            The new search space for that dimension.

        Raises
        ------
        ValueError
            If the key does not matched a pre-declared continuous dimension.
        """
        if not key in self._continuous_dims:
            valid_keys = self._continuous_dims.keys()
            raise ValueError(f'Unrecognized key {key} (expected one of {valid_keys})')
        self._continuous_dims[key] = val

    @property
    def discrete_dims(self) -> Dict[str, List[Any]]:
        """Describes the set of discrete parameters this control needs.

        Returns
        -------
        Dict[str, List[Any]]
            Will be of the form shape: parameter_name -> [value_1, value_2, ..., value_n]
        """
        return self._discrete_dims

    def update_discrete_dim(self, key: str, val: List[Any]):
        """Updates a specified discrete dimensions of the control with a
        user-provided value. Raises a ``ValueError`` if the key does not matched
        a pre-declared discrete control dimension.

        Parameters
        ----------
        key : str
            The key of the search dimension to override.
        val : List[Any]
            The new search space for that dimension.

        Raises
        ------
        ValueError
            If the key does not matched a pre-declared discrete dimension.
        """
        if not key in self._discrete_dims:
            valid_keys = self._discrete_dims.keys()
            raise ValueError(f'Unrecognized key {key} (expected one of {valid_keys})')
        self._discrete_dims[key] = val

    def __init__(self,
                 root_folder: str, *,
                 continuous_dims: Optional[Dict[str, Tuple[float, float]]] = None,
                 discrete_dims: Optional[Dict[str, List[Any]]] = None):
        """Construct a BaseControl

        Parameters
        ----------
        root_folder
            The folder containing all the data for this 3DB experiment. All
            paths are lative to his folder
        """
        self.root_folder = root_folder
        self._continuous_dims: Dict[str, Tuple[float, float]] = continuous_dims or {}
        self._discrete_dims: Dict[str, List[Any]] = discrete_dims or {}

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
        assert len(all_keys) == len(self.continuous_dims) + len(self.discrete_dims), \
            'Keys cannot be duplicated between continuous_dims and discrete_dims'
        if control_args.keys() != all_keys:
            return False, 'Keys in control arguments do not match declared keys'
        if not all(isinstance(control_args[k], float) or isinstance(control_args[k], int) for k in self.continuous_dims):
            return False, 'Some continuous arguments are not of type float'
        for k, v in self.discrete_dims.items():
            if not control_args[k] in v:
                return False, f'Argument {k} ({control_args[k]}) not in valid set: {v}'
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
