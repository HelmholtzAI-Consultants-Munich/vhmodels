"""Model interface, CLI and the Conda/Apptainer runtimes."""

from .base import BaseModel
from .factory import load_model

__all__ = ["BaseModel", "load_model"]
