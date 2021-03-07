import torch as ch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from threedb.rendering.utils import ControlsApplier
from typing import List

class RenderEnv(ABC): pass
class RenderObject(ABC): pass

class BaseRenderer(ABC):
    NAME: str = 'BaseRenderer'
    KEYS: List[str] = ['rgb']

    def __init__(self, 
                 root_dir: str, 
                 render_settings: Dict[str, Any],
                 env_extensions: List[str] = []) -> None:
        self.root_dir = root_dir
        self.env_extensions = env_extensions
        self.args = render_settings
    
    @staticmethod
    @abstractmethod
    def enumerate_models(search_dir: str) -> List[str]:
        """
        Given a root folder, returns all valid .blend files (according to a
        per-renderer convention).
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def enumerate_environments(search_dir: str) -> List[str]:
        """
        Given a root folder, returns all files in root/blender_environments/ which
        have extensions in ENV_EXTENSIONS above.
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
        in the abstract class :cla:`threedb.rendering.base_renderer.BaseRenderer`.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model: str) -> RenderObject:
        """
        Given a root directory and a model id, loads the model into the renderer
        and returns the corresponding object.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_model_uid(self, model: RenderObject) -> str:
        """
        Given an 3D model, return its UID as assigned by the renderer.
        """

    @abstractmethod
    def load_env(self, env: str) -> Optional[RenderEnv]:
        """
        Given a root folder and environment ID, load the environment into
        the renderer. If needed, returns an environment object, to be passed
        back to the render() function.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_render(self, 
                     model: Optional[RenderObject], 
                     env: Optional[RenderEnv]) -> None:
        """
        Perform setup operations for rendering. Called only when the model or
        environment being rendered changes---otherwise, only render() will be
        called. No return value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def render_and_apply(self, model_uid: str, object_class: int,
               applier: ControlsApplier, loaded_model: RenderObject,
               loaded_env: RenderEnv) -> Dict[str, ch.Tensor]:
        """
        Render a model and environment. You can assume that setup_render() has
        been called with the relevant model and object in context. This function
        should also handle applying the pre and post-processing controls.

        Arguments:
        - model_uid (str): The uid of the model assigned by the renderer.
        - object_class (int): The label (e.g., ImageNet class) of the object
          being rendered.
        - applier (ControlsApplier): An applier that we will use to apply the
          pre- and post-processing controls.
        - loaded_model: the model that was most recently loaded and passed to setup_render.
        - loaded_env: the environment that was most recently loaded and passed
          to setup_render.
        """
        raise NotImplementedError