import pkgutil
import importlib

# Import all modules in this package so VHModel subclasses register
package = __name__
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{package}.{module_name}")