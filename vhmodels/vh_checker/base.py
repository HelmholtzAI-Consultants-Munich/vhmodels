"""Interface every model implements: load_model, embed, predict, generate.

Imported inside model workers, so it must stay free of third-party dependencies.
"""

from abc import ABC, abstractmethod
import sys
import os
import importlib
from vhmodels.models import discovery


class BaseModel(ABC):
    @staticmethod
    def get_class(_class):
        class_path = discovery.find_class_path(_class)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "..", "models")

        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as error:
            raise ImportError(f"Error loading class {class_path}: {error}") from error

    @abstractmethod
    def load_model(self, model, resources, **kwargs):
        """Load ``model`` (a variant) from host-resolved ``resources`` ({name: dict})."""

    @abstractmethod
    def embed(self, input, **kwargs):
        pass

    @abstractmethod
    def predict(self, input, **kwargs):
        pass

    @abstractmethod
    def generate(self, input, **kwargs):
        pass
