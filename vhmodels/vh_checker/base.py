# The file contains the interface, which each model will implement:
# load_model, transform, predict, generate
from abc import ABC, abstractmethod
import sys
import os
import importlib
from vhmodels.registry import MODEL_REGISTRY

class BaseModel(ABC):
    
    @staticmethod
    def get_class(_class):
        class_path = MODEL_REGISTRY[_class]['class_path']

        # Determine the absolute path to your 'models' directory
        # Adjust '..' based on where runner.py sits relative to the models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "..", "models") # Example path
        
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        try:
            # Now Python can see 'DinoBloom' inside the models folder
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"Error loading class {class_path}: {e}", file=sys.stderr)
            sys.exit(1)
    
    @abstractmethod
    def load_model(self, model, **kwargs):
        pass
    
    @abstractmethod
    def transform(self, input, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, input, **kwargs):
        pass

    @abstractmethod
    def generate(self, input, **kwargs):
        pass