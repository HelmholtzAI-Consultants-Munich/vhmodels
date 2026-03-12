import pkgutil
import importlib
from .vh_checker.base import BaseModel
from .vh_checker.factory import load_model
from . import models

def discover_models():
    from . import models
    import pkgutil
    import importlib
    
    # walk_packages(path, prefix)
    # The prefix ensures we get 'vhmodels.models.hyformer' instead of just 'hyformer'
    for loader, module_name, is_pkg in pkgutil.walk_packages(models.__path__, models.__name__ + "."):
        
        # Scenario A: The file is directly inside models/ (e.g. models/test_model.py)
        # Scenario B: The file is inside a subfolder (e.g. models/Hyformer/model.py)
        if module_name.endswith(".model"):
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"Failed to load {module_name}: {e}")
        
        # Scenario C: If it's a sub-package (like 'vhmodels.models.hyformer'), 
        # we check if it has a 'model' submodule manually
        elif is_pkg:
            try:
                # Try to import the .model inside that package
                importlib.import_module(module_name + ".model")
            except ImportError:
                # This folder might not have a model.py, which is fine (e.g. __pycache__)
                continue

# Run discovery immediately
discover_models()

__all__ = ["load_model", "BaseModel"]