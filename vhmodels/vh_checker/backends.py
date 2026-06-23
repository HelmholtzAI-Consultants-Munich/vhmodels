# Runtime backends decide *how* the model runner (vhmodels.vh_checker.embed) is
# launched, e.g. inside a conda environment. ModelProxy is backend-agnostic:
# it builds the script arguments and delegates command construction, the
# availability check, and the subprocess environment to a RuntimeBackend.
#
# Only conda is implemented at the moment

from abc import ABC, abstractmethod

import os
import subprocess

# Repo root (…/virtual_human_chc), three levels up from this file.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# The runner module executed inside the isolated environment.
_RUNNER = ["python", "-m", "vhmodels.vh_checker.embed"]


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
        # Make the host source tree importable inside the env so `python -m
        # vhmodels.vh_checker.embed` resolves. This is superseded in a later
        # step, once `vh-checker create-env` installs vhmodels into the env and
        # the PYTHONPATH injection can be dropped entirely.
        env = os.environ.copy()
        env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        return env


def get_backend(runtime, env_name):
    """Return a RuntimeBackend for the requested runtime."""
    if runtime == "conda":
        return CondaBackend(env_name)
    raise NotImplementedError(
        f"Runtime '{runtime}' is not supported yet. Currently supported: conda."
    )
