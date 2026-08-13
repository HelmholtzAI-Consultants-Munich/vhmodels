"""Host-side lifecycle manager for persistent Apptainer model workers."""

import atexit
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
)


class ModelWorkerError(RuntimeError):
    """An exception raised by the loaded model inside a healthy worker."""


_MISSING_WORKER_MODULE = "No module named vhmodels.vh_checker.worker"


def _safe_instance_component(value):
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return component[:24] or "model"


class ApptainerProcessManager:
    """Own one Apptainer instance and its loaded, persistent model worker."""

    def __init__(
        self,
        backend,
        project,
        model,
        load_kwargs,
        timeout,
        run_subprocess,
        extract_frame,
    ):
        self.backend = backend
        self.project = project
        self.model = model
        self.load_kwargs = dict(load_kwargs)
        try:
            self.timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout must be a positive, finite number.") from error
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive, finite number.")
        self._run_subprocess = run_subprocess
        self._extract_frame = extract_frame

        self._creator_pid = os.getpid()
        self._project_name = _safe_instance_component(project)
        self._new_identity()

        self._lock = threading.RLock()
        self._instance_may_exist = False
        self._started = False
        self._closed = False
        self._atexit_callback = self._close_at_exit
        self._atexit_registered = False

    @property
    def is_started(self):
        with self._lock:
            return self._started

    def _new_identity(self):
        unique_suffix = f"{self._creator_pid}-{uuid.uuid4().hex[:12]}"
        self.instance_name = f"vhmodels-{self._project_name}-{unique_suffix}"
        self.socket_path = f"/tmp/{self.instance_name}.sock"

    def _register_atexit(self):
        if not self._atexit_registered:
            atexit.register(self._atexit_callback)
            self._atexit_registered = True

    def _unregister_atexit(self):
        if self._atexit_registered:
            atexit.unregister(self._atexit_callback)
            self._atexit_registered = False

    def _check_process(self):
        if os.getpid() != self._creator_pid:
            raise RuntimeError(
                "An Apptainer model proxy cannot be used after the process forks. "
                "Load a new proxy in the child process."
            )

    def _run(self, command, payload="", timeout=None):
        return self._run_subprocess(
            command,
            payload,
            self.backend.subprocess_env(),
            self.timeout if timeout is None else timeout,
        )

    def _request_locked(self, message):
        payload = json.dumps(message, ensure_ascii=False)
        connect_timeout = min(float(self.timeout), 30.0)
        command = self.backend.build_instance_request_command(
            self.instance_name,
            self.socket_path,
            connect_timeout,
        )
        stdout, stderr = self._run(command, payload)
        response = self._extract_frame(stdout, stderr)
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise ValueError(
                "The model worker returned a malformed response.\n"
                f"--- subprocess stdout ---\n{stdout}\n"
                f"--- subprocess stderr ---\n{stderr}"
            )
        if not response["ok"]:
            raise ModelWorkerError(
                response.get("error") or "Unknown model worker error."
            )
        return response.get("result")

    def _start_locked(self):
        if self._started:
            return
        if self._closed:
            raise RuntimeError("This model process manager is closed.")

        self.backend.prepare()
        command = self.backend.build_instance_start_command(
            self.instance_name, self.socket_path
        )
        # ``instance start`` detaches. If its CLI is interrupted after forking,
        # the UUID instance may exist even though the command never returned.
        self._instance_may_exist = True
        self._register_atexit()
        try:
            self._run(command, timeout=min(float(self.timeout), 60.0))
            self._request_locked(
                {
                    MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
                    "project": self.project,
                    "model": self.model,
                    "load_kwargs": self.load_kwargs,
                }
            )
        except BaseException as error:
            self._stop_locked(suppress_errors=True)
            if (
                isinstance(error, RuntimeError)
                and not isinstance(error, ModelWorkerError)
                and _MISSING_WORKER_MODULE in str(error)
            ):
                image_path = getattr(self.backend, "image_path", self.project)
                suggested_image = f"vhmodels-{self._project_name}-persistent.sif"
                raise RuntimeError(
                    f"The Apptainer image '{image_path}' was built before "
                    "persistent model workers were added and does not contain "
                    "'vhmodels.vh_checker.worker'. Rebuild it from the current "
                    "checkout. Existing images are not overwritten, so a safe "
                    "command is:\n"
                    f"  vh-checker create-apptainer-image {self.project} "
                    f"--output {suggested_image}\n"
                    "Then pass the new path as image_path=... to load_model()."
                ) from error
            raise
        self._started = True

    def embed(self, input, kwargs=None, cwd=None):
        """Send one request, starting and loading the worker if necessary."""
        self._check_process()
        validate_request_cwd = getattr(self.backend, "validate_request_cwd", None)
        if cwd is not None and validate_request_cwd is not None:
            validate_request_cwd(cwd)
        message = {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": input,
            "kwargs": kwargs or {},
            "cwd": os.fspath(Path(cwd).resolve()) if cwd is not None else None,
        }
        # Fail on non-JSON data before starting or invalidating a healthy model.
        json.dumps(message, ensure_ascii=False)

        with self._lock:
            self._check_process()
            if self._instance_may_exist and not self._started:
                raise RuntimeError(
                    "The previous Apptainer instance could not be stopped. "
                    "Call close() again before reusing this model."
                )
            self._start_locked()
            try:
                return self._request_locked(message)
            except ModelWorkerError:
                # A bad inference request does not make the loaded model or its
                # transport unhealthy; allow a later request to reuse it.
                raise
            except BaseException:
                # A timeout, dead relay, or malformed transport response leaves
                # worker health unknown. Stop it so the next request starts fresh.
                self._started = False
                self._stop_locked(suppress_errors=True)
                raise

    def _stop_locked(self, suppress_errors):
        if not self._instance_may_exist:
            self._started = False
            return True

        command = self.backend.build_instance_stop_command(self.instance_name)
        try:
            # Apptainer normally waits 10 seconds before force-killing an
            # instance; do not truncate that cleanup because inference uses a
            # smaller timeout.
            self._run(command, timeout=30.0)
        except BaseException:
            try:
                list_command = self.backend.build_instance_list_command(
                    self.instance_name
                )
                stdout, _ = self._run(list_command, timeout=30.0)
                still_exists = any(
                    columns and columns[0] == self.instance_name
                    for columns in (
                        line.split() for line in stdout.splitlines()
                    )
                )
            except BaseException:
                still_exists = True
            if still_exists:
                if not suppress_errors:
                    raise
                return False
        self._instance_may_exist = False
        self._started = False
        self._new_identity()
        self._unregister_atexit()
        return True

    def close(self):
        """Stop the instance once; calling close repeatedly is safe."""
        self._check_process()
        with self._lock:
            if self._closed:
                return
            self._check_process()
            self._stop_locked(suppress_errors=False)
            self._closed = True
        self._unregister_atexit()

    def _close_at_exit(self):
        if os.getpid() != self._creator_pid:
            return
        try:
            self.close()
        except BaseException:
            # Interpreter shutdown cannot report cleanup failures reliably.
            pass
