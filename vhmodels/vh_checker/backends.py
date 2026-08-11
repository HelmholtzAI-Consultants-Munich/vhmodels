"""Runtime strategies for launching the isolated model runner.

``ModelProxy`` delegates command construction, availability checks, runtime
preparation, and subprocess environment handling to these backends.
"""

from abc import ABC, abstractmethod

import os
import shutil
import subprocess
from pathlib import Path

from vhmodels.utils import lima_utils

# The runner module executed inside the isolated environment.
_RUNNER = ["python", "-m", "vhmodels.vh_checker.embed"]
_APPTAINER_RUNNER = ["/opt/venv/bin/python", "-m", "vhmodels.vh_checker.embed"]


class RuntimeBackend(ABC):
    """Strategy for launching the runner in an isolated environment."""

    @abstractmethod
    def build_command(self, script_args):
        """Return the full argv to run, given the runner's script arguments."""

    @abstractmethod
    def is_available(self):
        """Return True if this environment exists and can be used."""

    def subprocess_env(self):
        """Environment dict for the subprocess, or None to inherit the parent."""
        return None

    def is_runtime_available(self):
        """Return True if the runtime executable can be launched."""
        return True

    def prepare(self):
        """Prepare a runtime that needs one-time host setup."""


class CondaBackend(RuntimeBackend):
    def __init__(self, env_name):
        self.env_name = env_name

    def build_command(self, script_args):
        # --no-capture-output is REQUIRED: without it `conda run` does not
        # forward the parent's stdin to the runner, so the child reads empty
        # stdin and the model receives None. It also wires the child's
        # stdout/stderr straight to our pipes instead of conda buffering them.
        return (
            ["conda", "run", "--no-capture-output", "-n", self.env_name]
            + _RUNNER
            + list(script_args)
        )

    def is_available(self):
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True
        )
        return self.env_name in result.stdout

    def subprocess_env(self):
        # vhmodels is installed into the env by `vh-checker create-env`, so no
        # PYTHONPATH is needed for the runner or model discovery. Strip any
        # inherited PYTHONPATH so a host entry cannot shadow the env's packages.
        # Everything else (PATH, CUDA_VISIBLE_DEVICES, HF_HOME, ...) is passed
        # through.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        return env


class ApptainerBackend(RuntimeBackend):
    def __init__(self, image_path, use_lima=None):
        self.image_path = os.fspath(image_path)
        self.use_lima = lima_utils.uses_lima() if use_lima is None else use_lima

    def build_command(self, script_args):
        command = [
            "apptainer",
            "exec",
            self.image_path,
            *_APPTAINER_RUNNER,
            *list(script_args),
        ]
        if self.use_lima:
            return lima_utils.lima_shell_command(command, Path.cwd())
        return command

    def is_available(self):
        return Path(self.image_path).is_file()

    def is_runtime_available(self):
        executable = "limactl" if self.use_lima else "apptainer"
        return shutil.which(executable) is not None

    def prepare(self):
        if not self.use_lima:
            return
        if not lima_utils.is_lima_shared_path(self.image_path):
            raise RuntimeError(
                "On macOS, the Apptainer image must be under your home "
                "directory so Lima can access it."
            )
        if not lima_utils.is_lima_shared_path(Path.cwd()):
            raise RuntimeError(
                "On macOS, run vhmodels from a directory under your home "
                "directory so Lima can access relative input paths."
            )
        lima_utils.ensure_lima_instance()

    def subprocess_env(self):
        # The image contains its own vhmodels source and Python environment.
        # An inherited host PYTHONPATH could make the container import a
        # different checkout, while other useful settings (HF_HOME,
        # CUDA_VISIBLE_DEVICES, APPTAINER_BINDPATH, ...) should pass through.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("APPTAINERENV_PYTHONPATH", None)
        # Apptainer still accepts Singularity's legacy environment prefix.
        env.pop("SINGULARITYENV_PYTHONPATH", None)
        if self.use_lima:
            # Lima mounts the macOS home, but it is not the Linux guest's
            # $HOME. Bind it explicitly so absolute host input paths remain
            # visible inside the final Apptainer container.
            host_home = str(Path.home().resolve())
            bind_path = env.get("APPTAINER_BINDPATH")
            env["APPTAINER_BINDPATH"] = (
                f"{bind_path},{host_home}" if bind_path else host_home
            )
        return env


def get_backend(runtime, env_name):
    """Return a RuntimeBackend for the requested runtime."""
    if runtime == "conda":
        return CondaBackend(env_name)
    if runtime == "apptainer":
        return ApptainerBackend(env_name)
    raise NotImplementedError(
        f"Runtime '{runtime}' is not supported yet. Currently supported: conda, apptainer."
    )
