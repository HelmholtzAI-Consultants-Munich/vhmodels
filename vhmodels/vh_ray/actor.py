"""Ray actor class that loads and serves one model inside its per-model venv."""

import os
import socket
import sys

from vhmodels.vh_checker.base import BaseModel
from vhmodels.vh_checker.worker import _working_directory


def hide_paths(paths):
    """Remove directories from ``sys.path``, comparing resolved paths."""
    hidden = {os.path.realpath(path) for path in paths}
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.realpath(entry or os.curdir) not in hidden
    ]


class ModelActor:
    def __init__(self, project, model, resources, load_kwargs=None, driver_paths=()):
        # Ray prepends the driver's script dir and cwd to sys.path (see docs/ray.md).
        hide_paths(driver_paths)
        self._model = BaseModel.get_class(project)()
        self._model.load_model(model, resources, **(load_kwargs or {}))

    def embed(self, input, cwd=None, **kwargs):
        with _working_directory(cwd):
            return self._model.embed(input, **kwargs)

    def predict(self, input, embedding=None, cwd=None, **kwargs):
        with _working_directory(cwd):
            return self._model.predict(input, embedding=embedding, **kwargs)

    def info(self):
        """Describe the process the model runs in."""
        return {
            "executable": sys.executable,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sys_path": list(sys.path),
        }
