"""Runtime-specific transports for persistent model workers."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import subprocess
import tempfile

from vhmodels.vh_checker.protocol import send_request
from vhmodels.utils.subprocess_utils import terminate_process_group


class WorkerTransport(ABC):
    """Minimal lifecycle and request interface used by the process manager."""

    def __init__(self, backend):
        self.backend = backend

    def prepare(self):
        self.backend.prepare()

    def validate_request_cwd(self, cwd):
        validator = getattr(self.backend, "validate_request_cwd", None)
        if validator is not None:
            validator(cwd)

    @abstractmethod
    def start(self, worker_name, socket_path, timeout):
        """Start one persistent worker."""

    @abstractmethod
    def request(self, worker_name, socket_path, message, timeout):
        """Send one request and return its decoded response object."""

    @abstractmethod
    def stop(self, worker_name, socket_path):
        """Stop and reap the worker, raising if cleanup is not confirmed."""


class ApptainerWorkerTransport(WorkerTransport):
    """Manage an Apptainer instance and its in-container request relay."""

    def __init__(self, backend, run_subprocess, extract_frame):
        super().__init__(backend)
        self.run_subprocess = run_subprocess
        self.extract_frame = extract_frame

    def _run(self, command, payload="", timeout=30.0):
        return self.run_subprocess(
            command,
            payload,
            self.backend.subprocess_env(),
            timeout,
        )

    def start(self, worker_name, socket_path, timeout):
        command = self.backend.build_instance_start_command(worker_name, socket_path)
        self._run(command, timeout=min(timeout, 60.0))

    def request(self, worker_name, socket_path, message, timeout):
        command = self.backend.build_instance_request_command(
            worker_name,
            socket_path,
            min(timeout, 30.0),
        )
        stdout, stderr = self._run(
            command,
            json.dumps(message, ensure_ascii=False),
            timeout,
        )
        return self.extract_frame(stdout, stderr)

    def stop(self, worker_name, socket_path):
        command = self.backend.build_instance_stop_command(worker_name)
        try:
            self._run(command, timeout=30.0)
            return
        except BaseException as stop_error:
            try:
                list_command = self.backend.build_instance_list_command(worker_name)
                stdout, _ = self._run(list_command, timeout=30.0)
                still_exists = any(
                    columns and columns[0] == worker_name
                    for columns in (line.split() for line in stdout.splitlines())
                )
            except BaseException:
                still_exists = True
            if still_exists:
                raise stop_error


class CondaWorkerTransport(WorkerTransport):
    """Own one Conda worker process and connect directly to its Unix socket."""

    def __init__(
        self,
        backend,
        popen=None,
        request_sender=None,
        terminate_process=None,
    ):
        super().__init__(backend)
        self._popen = popen or subprocess.Popen
        self._request_sender = request_sender or send_request
        self._terminate_process = terminate_process or terminate_process_group
        self._process = None
        self._log = None

    def start(self, worker_name, socket_path, timeout):
        self._log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        command = self.backend.build_worker_start_command(socket_path)
        try:
            self._process = self._popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=self.backend.subprocess_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except BaseException:
            self._close_log()
            raise

    def _is_running(self):
        return self._process is not None and self._process.poll() is None

    def _diagnostics(self):
        if self._log is None:
            return ""
        self._log.flush()
        self._log.seek(0)
        output = self._log.read().strip()
        self._log.seek(0, 2)
        return f"\n--- worker output ---\n{output}" if output else ""

    def request(self, worker_name, socket_path, message, timeout):
        try:
            return self._request_sender(
                socket_path,
                message,
                min(timeout, 30.0),
                timeout,
                self._is_running,
            )
        except Exception as error:
            if not self._is_running():
                raise RuntimeError(
                    "The Conda model worker exited unexpectedly." + self._diagnostics()
                ) from error
            if isinstance(error, TimeoutError):
                raise RuntimeError(
                    f"The Conda model worker request exceeded {timeout:g}s."
                ) from error
            raise

    def stop(self, worker_name, socket_path):
        if self._process is not None:
            self._terminate_process(self._process)
            self._process = None
        try:
            Path(socket_path).unlink()
        except FileNotFoundError:
            pass
        self._close_log()

    def _close_log(self):
        if self._log is not None:
            self._log.close()
            self._log = None
