"""Host-side model proxy and subprocess handover protocol."""

import json
import os
import signal
import subprocess
from pathlib import Path

from vhmodels.registry import MODEL_REGISTRY
from vhmodels.vh_checker.backends import get_backend
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    RESULT_MARKER,
)

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
        self,
        project,
        env_name,
        model=None,
        runtime="conda",
        timeout=DEFAULT_TIMEOUT,
        load_kwargs=None,
    ):
        self.project = project
        self.model = model
        self.runtime = runtime
        self.env_name = env_name
        self.timeout = timeout
        self.load_kwargs = load_kwargs or {}
        # Selecting the backend here fails fast on an unsupported runtime.
        self.backend = get_backend(runtime, env_name)

    def embed(self, input, **kwargs):
        if not self.backend.is_available():
            if self.runtime == "apptainer":
                raise RuntimeError(
                    f"The Apptainer image '{self.env_name}' does not exist. "
                    f"Please run 'vh-checker create-apptainer-image "
                    f"{self.project}' first."
                )
            raise RuntimeError(
                f"The environment '{self.env_name}' does not exist. "
                f"Please run 'vh-checker create-env {self.project}' first."
            )
        if not self.backend.is_runtime_available():
            if getattr(self.backend, "use_lima", False):
                raise RuntimeError(
                    "Lima is not available. On macOS, install it with "
                    "'brew install lima' and ensure 'limactl' is in PATH."
                )
            raise RuntimeError(
                "The Apptainer executable is not available. Install Apptainer "
                "and ensure 'apptainer' is in PATH."
            )
        self.backend.prepare()

        # The child reads one tagged JSON message per line from stdin.
        payload = "\n".join(
            [
                json.dumps(
                    {
                        MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
                        "load_kwargs": self.load_kwargs,
                    }
                ),
                json.dumps({MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": input}),
            ]
        )

        script_args = ["--project", self.project]
        if self.model:
            script_args += ["--model", self.model]

        cmd = self.backend.build_command(script_args)
        stdout, stderr = _run_subprocess(
            cmd, payload, self.backend.subprocess_env(), self.timeout
        )
        return _extract_result(stdout, stderr)


def load_model(project, model=None, runtime="conda", image_path=None, **load_kwargs):
    if project not in list(MODEL_REGISTRY.keys()):
        raise ValueError(f"Model '{project}' not found.")

    env_name = "vhmodels-" + project
    if runtime == "apptainer":
        # Match the default output of ``create-apptainer-image``. Resolve the
        # path now so changing the working directory between load and embed
        # cannot silently select a different image.
        image_path = image_path or f"{env_name}.sif"
        env_name = str(Path(image_path).expanduser().resolve())
    elif image_path is not None:
        raise ValueError("image_path can only be used with runtime='apptainer'.")

    return ModelProxy(
        project=project,
        env_name=env_name,
        model=model,
        runtime=runtime,
        load_kwargs=load_kwargs,
    )
