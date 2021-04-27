"""
threedb.evaluators.base_evaluator
==================================

Provides an abstract base class for implementing evaluators.

See :class:`threedb.evaluators.classification.SimpleClassificationEvaluator` for further details.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any

from torch import Tensor

Output = Union[bool, int, float, str]
LabelType = Union[List[int], int]

class BaseEvaluator(ABC):
    """Abstract Base Class for Evaluators

    An evaluator is in charge of taking the output of an inference model,
    analyzing it, and returning the relevant summary outputs (e.g.,
    correctness for classification, IoU for detection, MSE for segmentation,
    etc.)
    """
    KEYS: List[str] = ['is_correct', 'loss']

    @abstractmethod
    def get_segmentation_label(self, model_uid: str) -> int:
        """Given a model_uid, return a scalar label corresponding to the 3D
        model. This label is only used for the purpopses of generating a
        segmentation map. If the only goal is to separate the object from its
        background, this function can return anything greater than zero.

        Parameters
        ----------
        model_uid : str
            Unique identifier for the model.

        Returns
        -------
        int
            An integer which will be the color of the model in the segmentation
            map.
        """
        raise NotImplementedError

    @abstractmethod
    def get_target(self, model_uid: str, render_output: Dict[str, Tensor]) -> Any:
        """Given a model_uid, returns the corresponding label for that 3D model,
        which will be used to evaluate accuracy, loss, and other statistics.

        Parameters
        ----------
        model_uid : str
            The unique id of the model, as returned by (:meth:`threedb.rendering.base_renderer.BaseRenderer.render_and_apply`)
            (for blender, this will be the object ID)

        Returns
        -------
        int
            The class label for that object, most likely looked
            up through the evaluator metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def summary_stats(self, pred: Any, label: Any) -> Dict[str, Output]:
        """Given a tensor corresponding to a model prediction on a single
        rendered image, and a tensor containing the label for that image, return
        a dictionary of summary statistics. The keys of the dictionary should
        match the ``KEYS`` static property of the concrete class.

        Parameters
        ----------
        pred : Any
            The output of the model being inspected with 3DB. [TODO]
        label : Any
            The label corresponding to the rendered image.

        Returns
        -------
        Dict[str, Output]
            A dictionary mapping each key declared in ``type(self).KEYS`` to a
            serializable value (``int``, ``str``, ``float``, or ``bool``).
        """
        raise NotImplementedError

    @abstractmethod
    def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
        """
        This function declares what the output of render() will be, based on the
        renderer settings. Returns a dictionary mapping keys to (dtype, size)
        tuples---the output of render() is string-to-tensor dictionary whose
        tensors will be checked against the return value of this function for
        both size and type.

        A basic implementation which suffices for most applications is provided
        in the abstract class :class:`threedb.rendering.base_renderer.BaseRenderer`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_tensor(pred: Any, output_shape: List[int], input_shape: List[int]) -> Tensor:
        """Turns the output of the inference model into a PyTorch tensor. Useful
        for, e.g., detection models, where the model typically outputs a
        dictionary.

        Parameters
        ----------
        pred : Any
            The output of the inference model.
        output_shape : List[int]
            The desired shape of the output tensor
        input_shape : List[int]
            The shape of the image inputted into the inference model.

        Returns
        -------
        Tensor
            a Tensor representation of the model output
        """
        raise NotImplementedError
