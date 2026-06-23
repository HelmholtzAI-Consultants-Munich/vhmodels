# This file contains the wrapper class ModelProxy
# It controls what arguments are passed to the runner file and owns the
# parent side of the parent<->child handover protocol (see protocol.py).

from vhmodels.registry import MODEL_REGISTRY
from vhmodels.vh_checker.protocol import RESULT_MARKER

import os
import json
import signal
import subprocess

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
env = os.environ.copy()
env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

DEFAULT_TIMEOUT = 600


def _terminate_group(proc):
    """SIGTERM then SIGKILL the child's entire process group, then reap it.

    ``conda run`` (and container launchers) spawn the real python interpreter as
    a grandchild, so killing only the immediate child would orphan it (and any
    GPU memory it holds). Because the child is started in its own session via
    ``start_new_session=True``, every descendant shares its process group and is
    reached by ``os.killpg``.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return  # process already gone
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue  # escalate to SIGKILL


def _run_subprocess(cmd, payload, subprocess_env, timeout):
    """Run ``cmd``, send ``payload`` on stdin, return (stdout, stderr).

    Raises RuntimeError on timeout or non-zero exit, including both streams.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=subprocess_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,  # setsid(): child leads its own group+session
    )
    try:
        stdout, stderr = proc.communicate(input=payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_group(proc)  # kill launcher + grandchild model process
        stdout, stderr = proc.communicate()  # drain anything buffered
        raise RuntimeError(
            f"Model subprocess exceeded {timeout}s and was killed.\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Model subprocess failed (exit {proc.returncode}).\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    return stdout, stderr


def _extract_result(stdout, stderr):
    """Extract the framed JSON result from the child's stdout.

    Requires both the opening and closing RESULT_MARKER; a missing closing
    marker means the output was truncated (e.g. the child was killed mid-write).
    Returns the value under the model's "output" key (the established contract).
    """

    def _fail(reason):
        raise ValueError(
            f"{reason}\n"
            f"--- subprocess stdout ---\n{stdout}\n"
            f"--- subprocess stderr ---\n{stderr}"
        )

    open_idx = stdout.find(RESULT_MARKER)
    if open_idx == -1:
        _fail("No result marker found in subprocess output (no opening marker).")

    start = open_idx + len(RESULT_MARKER)
    close_idx = stdout.find(RESULT_MARKER, start)
    if close_idx == -1:
        _fail("Result truncated: opening marker present but closing marker missing.")

    chunk = stdout[start:close_idx]
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError as e:
        _fail(f"Result frame is not valid JSON ({e}).")

    if "output" not in parsed:
        _fail(f"Model result missing 'output' key: {parsed!r}")
    return parsed["output"]


class ModelProxy:
    def __init__(
        self, project, env_name, model=None, runtime="conda", timeout=DEFAULT_TIMEOUT
    ):
        self.project = project
        self.model = model
        self.runtime = runtime
        self.env_name = env_name
        self.timeout = timeout

    def _env_exists(self):
        # A quick way to check if a conda env exists
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True
        )
        return self.env_name in result.stdout

    def embed(self, input, **kwargs):
        if not self._env_exists():
            raise RuntimeError(
                f"The environment '{self.env_name}' does not exist. "
                f"Please run 'vh-checker create-env {self.project}' first."
            )

        # The whole input is serialized to JSON and sent on the child's stdin.
        # File-path inputs travel as JSON strings too; the model resolves them.
        payload = json.dumps(input)

        # Run using conda env.
        if self.runtime == "conda":
            cmd = [
                "conda",
                "run",
                "-n",
                self.env_name,
                "python",
                "-m",
                "vhmodels.vh_checker.embed",
                "--project",
                self.project,
            ]
        elif self.runtime == "docker":
            cmd = [
                "docker",
                "run",
                "--rm",
                "-i",
                f"vhmodels-{self.project}",
                "micromamba",
                "run",
                "-n",
                f"vhmodels-{self.project}",
                "python",
                "-m",
                "vhmodels.vh_checker.embed",
                "--project",
                self.project,
            ]
        elif self.runtime == "singularity":
            raise NotImplementedError("Singularity not implemented yet!")
        else:
            raise RuntimeError(
                f"Unsupported runtime: '{self.runtime}'. "
                f"Supported environments: conda, docker, singularity."
            )

        if hasattr(self, "model") and self.model:
            cmd.extend(["--model", self.model])

        stdout, stderr = _run_subprocess(cmd, payload, env, self.timeout)
        return _extract_result(stdout, stderr)


def load_model(project, model=None, runtime="conda"):
    if project not in list(MODEL_REGISTRY.keys()):
        raise ValueError(f"Model '{project}' not found.")

    return ModelProxy(
        project=project, env_name="vhmodels-" + project, model=model, runtime=runtime
    )
