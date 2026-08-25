"""Host-side lifecycle managers for persistent model workers."""

import atexit
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import uuid

from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    PREDICT_MESSAGE_TYPE,
)
from vhmodels.vh_checker.transports import (
    ApptainerWorkerTransport,
    CondaWorkerTransport,
)


class ModelWorkerError(RuntimeError):
    """An exception raised by the loaded model inside a healthy worker."""


_MISSING_WORKER_MODULE = "No module named vhmodels.vh_checker.worker"


def _log(tag, message):
    print(f"[{tag}] {message}", file=sys.stderr)


def _safe_name_component(value):
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return component[:24] or "model"


class ModelProcessManager:
    """Load one model in a persistent worker reached through a transport."""

    def __init__(
        self,
        transport,
        project,
        model,
        load_kwargs,
        timeout,
        runtime_name,
    ):
        self.transport = transport
        self.backend = transport.backend
        self.project = project
        self.model = model
        self.load_kwargs = dict(load_kwargs)
        try:
            self.timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout must be a positive, finite number.") from error
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive, finite number.")
        self.runtime_name = runtime_name

        self._creator_pid = os.getpid()
        self._project_name = _safe_name_component(project)
        self._new_identity()

        self._lock = threading.RLock()
        self._worker_may_exist = False
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
        self.worker_name = f"vhmodels-{self._project_name}-{unique_suffix}"
        # Keep the Apptainer-specific public name for compatibility with
        # lifecycle tooling and existing callers that inspect their instance.
        self.instance_name = self.worker_name
        self.socket_path = f"/tmp/{self.worker_name}.sock"

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
                "A model proxy cannot be used after the process forks. "
                "Load a new proxy in the child process."
            )

    @staticmethod
    def _unwrap_response(response):
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise ValueError(
                f"The model worker returned a malformed response: {response!r}"
            )
        if not response["ok"]:
            raise ModelWorkerError(
                response.get("error") or "Unknown model worker error."
            )
        return response.get("result")

    def _request_locked(self, message):
        response = self.transport.request(
            self.worker_name,
            self.socket_path,
            message,
            self.timeout,
        )
        return self._unwrap_response(response)

    def _translate_start_error(self, error):
        return None

    def _start_locked(self):
        if self._started:
            return
        if self._closed:
            raise RuntimeError("This model process manager is closed.")

        self.transport.prepare()
        _log("BACKEND", f"{self.runtime_name} runtime ready")
        # A launcher may create its worker before reporting an error, so mark
        # ownership before starting and make the exact identity safe to stop.
        self._worker_may_exist = True
        self._register_atexit()
        try:
            self.transport.start(self.worker_name, self.socket_path, self.timeout)
            _log("PROCESS MANAGER", f"{self.runtime_name} worker started")
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
            translated = self._translate_start_error(error)
            if translated is not None:
                raise translated from error
            raise
        self._started = True
        variant = f" ({self.model})" if self.model is not None else ""
        # TODO: move somehow to worker-side logging
        _log("REGISTRY", f"Model manifest '{self.project}'{variant} resolved")
        _log("RESOURCES", f"Local resources resolved for '{self.project}'{variant}")
        _log("MODEL WORKER", f"Model '{self.project}'{variant} fully loaded")

    def _dispatch(self, message_type, log_verb, kwargs, cwd, **extra_fields):
        """Send one request, starting and loading the worker if necessary."""
        self._check_process()
        if cwd is not None:
            self.transport.validate_request_cwd(cwd)
        message = {
            MESSAGE_TYPE_KEY: message_type,
            "kwargs": kwargs or {},
            "cwd": os.fspath(Path(cwd).resolve()) if cwd is not None else None,
            **extra_fields,
        }
        # Fail on non-JSON data before starting or invalidating a healthy model.
        json.dumps(message, ensure_ascii=False)

        with self._lock:
            self._check_process()
            if self._worker_may_exist and not self._started:
                raise RuntimeError(
                    f"The previous {self.runtime_name} worker could not be stopped. "
                    "Call close() again before reusing this model."
                )
            self._start_locked()
            _log("MODEL", f"Starting '{self.project}' {log_verb}...")
            try:
                return self._request_locked(message)
            except ModelWorkerError:
                # A rejected request does not invalidate an otherwise healthy
                # loaded model, so allow a later request to reuse the worker.
                raise
            except BaseException:
                self._started = False
                self._stop_locked(suppress_errors=True)
                raise

    def embed(self, input, kwargs=None, cwd=None):
        return self._dispatch(
            EMBED_MESSAGE_TYPE, 
            "embedding", 
            kwargs, 
            cwd, 
            input=input
        )

    def predict(self, input, embedding, kwargs=None, cwd=None):
        return self._dispatch(
            PREDICT_MESSAGE_TYPE,
            "prediction",
            kwargs,
            cwd,
            input=input,
            embedding=embedding,
        )

    def _stop_locked(self, suppress_errors):
        if not self._worker_may_exist:
            self._started = False
            return True

        try:
            self.transport.stop(self.worker_name, self.socket_path)
        except BaseException:
            if not suppress_errors:
                raise
            return False

        self._worker_may_exist = False
        self._started = False
        self._new_identity()
        self._unregister_atexit()
        return True

    def close(self):
        """Stop the worker once; calling close repeatedly is safe."""
        self._check_process()
        with self._lock:
            if self._closed:
                return
            self._check_process()
            self._stop_locked(suppress_errors=False)
            self._closed = True
        self._unregister_atexit()
        _log("MODEL", f"Model '{self.project}' instance closed")

    def _close_at_exit(self):
        if os.getpid() != self._creator_pid:
            return
        try:
            self.close()
        except BaseException:
            pass


class ApptainerProcessManager(ModelProcessManager):
    """Persistent manager using an Apptainer instance transport."""

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
        # Resolve through attributes so existing test/application injection can
        # replace the subprocess helpers after construction.
        self._run_subprocess = run_subprocess
        self._extract_frame = extract_frame
        transport = ApptainerWorkerTransport(
            backend,
            run_subprocess=lambda *args, **kwargs: self._run_subprocess(
                *args, **kwargs
            ),
            extract_frame=lambda *args, **kwargs: self._extract_frame(*args, **kwargs),
        )
        super().__init__(
            transport=transport,
            project=project,
            model=model,
            load_kwargs=load_kwargs,
            timeout=timeout,
            runtime_name="Apptainer",
        )

    def _translate_start_error(self, error):
        if (
            isinstance(error, RuntimeError)
            and not isinstance(error, ModelWorkerError)
            and _MISSING_WORKER_MODULE in str(error)
        ):
            image_path = getattr(self.backend, "image_path", self.project)
            suggested_image = f"vhmodels-{self._project_name}-persistent.sif"
            return RuntimeError(
                f"The Apptainer image '{image_path}' was built before "
                "persistent model workers were added and does not contain "
                "'vhmodels.vh_checker.worker'. Rebuild it from the current "
                "checkout. Existing images are not overwritten, so a safe "
                "command is:\n"
                f"  vh-checker create-apptainer-image {self.project} "
                f"--output {suggested_image}\n"
                "Then pass the new path as image_path=... to load_model()."
            )
        return None


class CondaProcessManager(ModelProcessManager):
    """Persistent manager using one directly connected Conda process."""

    def __init__(
        self,
        backend,
        project,
        model,
        load_kwargs,
        timeout,
        popen=None,
        request_sender=None,
        terminate_process=None,
    ):
        transport = CondaWorkerTransport(
            backend,
            popen=popen,
            request_sender=request_sender,
            terminate_process=terminate_process,
        )
        super().__init__(
            transport=transport,
            project=project,
            model=model,
            load_kwargs=load_kwargs,
            timeout=timeout,
            runtime_name="Conda",
        )
