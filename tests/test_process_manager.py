"""Tests for persistent, model-agnostic Conda and Apptainer workers."""

import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from vhmodels.vh_checker import factory, process_manager, worker
from vhmodels.vh_checker.process_manager import (
    ApptainerProcessManager,
    CondaProcessManager,
    ModelWorkerError,
)
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
)


class FakeBackend:
    """Record lifecycle command construction without invoking Apptainer."""

    def __init__(self):
        self.prepare_calls = 0
        self.environment = {"FAKE_APPTAINER_ENV": "present"}
        self.image_path = "/images/fake-model.sif"

    def prepare(self):
        self.prepare_calls += 1

    def subprocess_env(self):
        return self.environment

    def validate_request_cwd(self, cwd):
        return None

    def build_instance_start_command(self, instance_name, socket_path):
        return ("start", instance_name, socket_path)

    def build_instance_request_command(
        self, instance_name, socket_path, connect_timeout
    ):
        return ("request", instance_name, socket_path, connect_timeout)

    def build_instance_stop_command(self, instance_name):
        return ("stop", instance_name)

    def build_instance_list_command(self, instance_name):
        return ("list", instance_name)


class FakeCondaBackend:
    def __init__(self):
        self.env_name = "vhmodels-generic-model"
        self.prepare_calls = 0
        self.environment = {"FAKE_CONDA_ENV": "present"}

    def prepare(self):
        self.prepare_calls += 1

    def subprocess_env(self):
        return self.environment

    def build_worker_start_command(self, socket_path):
        return ("conda-worker", self.env_name, socket_path)


class FakeCondaProcess:
    _next_pid = 9000

    def __init__(self):
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.returncode = None

    def poll(self):
        return self.returncode


class FakeTransport:
    """Act like the lightweight worker relay while retaining every message."""

    def __init__(self):
        self.calls = []
        self.messages = []
        self.start_count = 0
        self.stop_count = 0
        self.load_count = 0
        self.embed_count = 0
        self.model_errors = {}
        self.transport_failures = set()
        self.missing_instances = set()

    @staticmethod
    def extract_frame(stdout, stderr):
        return json.loads(stdout)

    def run(self, command, payload, environment, timeout):
        self.calls.append(
            {
                "command": command,
                "payload": payload,
                "environment": environment,
                "timeout": timeout,
            }
        )
        operation = command[0]
        if operation == "start":
            self.start_count += 1
            return "", ""
        if operation == "stop":
            self.stop_count += 1
            return "", ""
        if operation == "list":
            instance_name = command[1]
            row = "" if instance_name in self.missing_instances else instance_name
            return f"INSTANCE NAME PID IMAGE\n{row}\n", ""

        assert operation == "request"
        message = json.loads(payload)
        self.messages.append(message)
        message_type = message[MESSAGE_TYPE_KEY]
        if message_type == LOAD_MESSAGE_TYPE:
            self.load_count += 1
            response = {"ok": True, "result": None}
        else:
            assert message_type == EMBED_MESSAGE_TYPE
            self.embed_count += 1
            request_number = self.embed_count
            if request_number in self.transport_failures:
                raise RuntimeError("relay exited before returning a frame")
            if request_number in self.model_errors:
                response = {
                    "ok": False,
                    "error": self.model_errors[request_number],
                }
            else:
                response = {
                    "ok": True,
                    "result": {
                        "output": {
                            "input": message["input"],
                            "kwargs": message["kwargs"],
                            "cwd": message["cwd"],
                        }
                    },
                }
        return json.dumps(response, ensure_ascii=False), ""


class MissingWorkerTransport(FakeTransport):
    """Simulate an image created before the persistent worker was packaged."""

    def run(self, command, payload, environment, timeout):
        if command[0] == "request":
            raise RuntimeError(
                "Model subprocess failed (exit 1).\n"
                "/opt/venv/bin/python: No module named "
                "vhmodels.vh_checker.worker"
            )
        return super().run(command, payload, environment, timeout)


@pytest.fixture
def make_manager():
    created = []

    def make(
        *,
        project="generic-model",
        model="variant-a",
        load_kwargs=None,
        timeout=15,
        backend=None,
        transport=None,
    ):
        backend = backend or FakeBackend()
        transport = transport or FakeTransport()
        manager = ApptainerProcessManager(
            backend=backend,
            project=project,
            model=model,
            load_kwargs=load_kwargs or {},
            timeout=timeout,
            run_subprocess=transport.run,
            extract_frame=transport.extract_frame,
        )
        created.append(manager)
        return manager, backend, transport

    yield make

    # Also unregister every atexit hook. Individual tests still make their own
    # lifecycle assertions before this safety-net cleanup runs.
    for manager in reversed(created):
        try:
            manager.close()
        except Exception:
            pass


def test_model_worker_loads_once_and_handles_arbitrary_requests(monkeypatch, tmp_path):
    calls = []

    class RecordingModel:
        def __init__(self):
            calls.append(("init",))

        def load_model(self, model_name, **kwargs):
            calls.append(("load", model_name, kwargs))

        def embed(self, input, **kwargs):
            calls.append(("embed", input, kwargs, Path.cwd()))
            return {
                "output": {
                    "echo": input,
                    "kwargs": kwargs,
                    "cwd": os.fspath(Path.cwd()),
                }
            }

    monkeypatch.setattr(
        worker.BaseModel,
        "get_class",
        staticmethod(
            lambda project: calls.append(("class", project)) or RecordingModel
        ),
    )

    dispatcher = worker.ModelWorker()
    assert (
        dispatcher.handle(
            {
                MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
                "project": "any-registry-entry",
                "model": "any-variant",
                "load_kwargs": {"device": "cpu", "precision": "float32"},
            }
        )
        is None
    )

    caller_cwd = Path.cwd()
    request_cwd = tmp_path / "data directory"
    request_cwd.mkdir()
    nested_input = {
        "records": [1, 2.5, None, True, {"unicode": "München ⚕"}],
        "metadata": {"line": "first\nsecond"},
    }
    first_kwargs = {"batch_size": 7, "options": {"normalize": True}}

    first_result = dispatcher.handle(
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": nested_input,
            "kwargs": first_kwargs,
            "cwd": os.fspath(request_cwd),
        }
    )
    second_result = dispatcher.handle(
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": ["another", {"shape": [2, 3]}],
            "kwargs": {"return_mask": False},
            "cwd": None,
        }
    )

    assert first_result == {
        "output": {
            "echo": nested_input,
            "kwargs": first_kwargs,
            "cwd": os.fspath(request_cwd),
        }
    }
    assert second_result["output"]["echo"] == ["another", {"shape": [2, 3]}]
    assert Path.cwd() == caller_cwd
    assert calls[:3] == [
        ("class", "any-registry-entry"),
        ("init",),
        (
            "load",
            "any-variant",
            {"device": "cpu", "precision": "float32"},
        ),
    ]
    assert [call[0] for call in calls].count("load") == 1
    assert [call[0] for call in calls].count("embed") == 2

    with pytest.raises(RuntimeError, match="already has a loaded model"):
        dispatcher.handle(
            {
                MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
                "project": "a-different-project",
            }
        )
    assert [call[0] for call in calls].count("load") == 1


def test_model_worker_remains_loaded_after_model_error(monkeypatch):
    calls = {"load": 0, "embed": 0}

    class FlakyModel:
        def load_model(self, model_name, **kwargs):
            calls["load"] += 1

        def embed(self, input, **kwargs):
            calls["embed"] += 1
            if input == "bad input":
                raise ValueError("input rejected by model")
            return {"output": input}

    monkeypatch.setattr(
        worker.BaseModel,
        "get_class",
        staticmethod(lambda project: FlakyModel),
    )
    dispatcher = worker.ModelWorker()
    dispatcher.handle(
        {
            MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
            "project": "generic-model",
        }
    )

    with pytest.raises(ValueError, match="input rejected by model"):
        dispatcher.handle({MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": "bad input"})

    assert dispatcher.is_loaded is True
    assert dispatcher.handle(
        {MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": "valid input"}
    ) == {"output": "valid input"}
    assert calls == {"load": 1, "embed": 2}


def test_conda_manager_starts_once_requests_directly_and_reaps(tmp_path):
    backend = FakeCondaBackend()
    processes = []
    popen_calls = []
    terminated = []
    messages = []

    def fake_popen(command, **kwargs):
        process = FakeCondaProcess()
        processes.append(process)
        popen_calls.append((command, kwargs))
        return process

    def fake_request(socket_path, message, connect_timeout, timeout, is_running):
        assert is_running()
        messages.append(message)
        result = None
        if message[MESSAGE_TYPE_KEY] == EMBED_MESSAGE_TYPE:
            result = {
                "output": {
                    "input": message["input"],
                    "kwargs": message["kwargs"],
                }
            }
        return {"ok": True, "result": result}

    def fake_terminate(process):
        process.returncode = -15
        terminated.append(process)

    manager = CondaProcessManager(
        backend=backend,
        project="generic-model",
        model="variant-a",
        load_kwargs={"device": "cpu"},
        timeout=12,
        popen=fake_popen,
        request_sender=fake_request,
        terminate_process=fake_terminate,
    )

    first = manager.embed("first", kwargs={"batch_size": 2}, cwd=tmp_path)
    second = manager.embed({"nested": [1, None]}, kwargs={"normalize": True})

    assert first == {"output": {"input": "first", "kwargs": {"batch_size": 2}}}
    assert second["output"]["input"] == {"nested": [1, None]}
    assert backend.prepare_calls == 1
    assert len(processes) == 1
    command, popen_kwargs = popen_calls[0]
    assert command == (
        "conda-worker",
        "vhmodels-generic-model",
        manager.socket_path,
    )
    assert popen_kwargs["stdin"] is subprocess.DEVNULL
    assert popen_kwargs["stderr"] is subprocess.STDOUT
    assert popen_kwargs["env"] is backend.environment
    assert popen_kwargs["start_new_session"] is True
    assert [message[MESSAGE_TYPE_KEY] for message in messages] == [
        LOAD_MESSAGE_TYPE,
        EMBED_MESSAGE_TYPE,
        EMBED_MESSAGE_TYPE,
    ]
    assert messages[0] == {
        MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
        "project": "generic-model",
        "model": "variant-a",
        "load_kwargs": {"device": "cpu"},
    }

    manager.close()
    manager.close()
    assert terminated == processes


def test_conda_transport_failure_restarts_and_reloads():
    backend = FakeCondaBackend()
    processes = []
    terminated = []
    messages = []
    embed_count = 0

    def fake_popen(command, **kwargs):
        process = FakeCondaProcess()
        processes.append(process)
        return process

    def flaky_request(socket_path, message, connect_timeout, timeout, is_running):
        nonlocal embed_count
        messages.append(message)
        if message[MESSAGE_TYPE_KEY] == EMBED_MESSAGE_TYPE:
            embed_count += 1
            if embed_count == 1:
                raise TimeoutError("worker did not answer")
            return {"ok": True, "result": {"output": message["input"]}}
        return {"ok": True, "result": None}

    def fake_terminate(process):
        process.returncode = -15
        terminated.append(process)

    manager = CondaProcessManager(
        backend=backend,
        project="generic-model",
        model=None,
        load_kwargs={},
        timeout=12,
        popen=fake_popen,
        request_sender=flaky_request,
        terminate_process=fake_terminate,
    )

    with pytest.raises(RuntimeError, match="request exceeded 12s"):
        manager.embed("first")
    assert not manager.is_started

    assert manager.embed("second") == {"output": "second"}
    assert len(processes) == 2
    assert terminated == [processes[0]]
    assert [message[MESSAGE_TYPE_KEY] for message in messages].count(
        LOAD_MESSAGE_TYPE
    ) == 2

    manager.close()
    assert terminated == processes


def test_conda_worker_startup_crash_reports_output_and_cleans_up():
    backend = FakeCondaBackend()
    process = FakeCondaProcess()
    process.returncode = 1
    terminated = []

    def failed_popen(command, **kwargs):
        kwargs["stdout"].write("worker import failed")
        kwargs["stdout"].flush()
        return process

    def no_request(socket_path, message, connect_timeout, timeout, is_running):
        assert not is_running()
        raise RuntimeError("worker did not become ready")

    manager = CondaProcessManager(
        backend=backend,
        project="generic-model",
        model=None,
        load_kwargs={},
        timeout=12,
        popen=failed_popen,
        request_sender=no_request,
        terminate_process=lambda child: terminated.append(child),
    )

    with pytest.raises(RuntimeError, match="worker import failed"):
        manager.embed("request")

    assert terminated == [process]
    assert not manager.is_started


def test_manager_starts_and_loads_once_for_many_requests(make_manager, tmp_path):
    manager, backend, transport = make_manager(
        project="generic-model",
        model="variant-z",
        load_kwargs={"device": "cpu", "revision": "v2"},
    )
    request_cwd = tmp_path / "inputs"
    request_cwd.mkdir()
    first_input = {"items": [1, None, {"name": "naïve"}]}

    first = manager.embed(
        first_input,
        kwargs={"batch_size": 4},
        cwd=request_cwd,
    )
    second = manager.embed(
        ["x", {"coordinates": [1.25, -3]}],
        kwargs={"normalize": True},
    )

    assert first == {
        "output": {
            "input": first_input,
            "kwargs": {"batch_size": 4},
            "cwd": os.fspath(request_cwd.resolve()),
        }
    }
    assert second["output"]["input"] == ["x", {"coordinates": [1.25, -3]}]
    assert manager.is_started is True
    assert backend.prepare_calls == 1
    assert transport.start_count == 1
    assert transport.load_count == 1
    assert transport.embed_count == 2
    assert transport.messages == [
        {
            MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
            "project": "generic-model",
            "model": "variant-z",
            "load_kwargs": {"device": "cpu", "revision": "v2"},
        },
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": first_input,
            "kwargs": {"batch_size": 4},
            "cwd": os.fspath(request_cwd.resolve()),
        },
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": ["x", {"coordinates": [1.25, -3]}],
            "kwargs": {"normalize": True},
            "cwd": None,
        },
    ]
    assert all(call["environment"] is backend.environment for call in transport.calls)


def test_model_error_keeps_loaded_manager_reusable(make_manager):
    transport = FakeTransport()
    transport.model_errors[1] = "ValueError: unsupported record"
    manager, backend, _ = make_manager(transport=transport)

    with pytest.raises(ModelWorkerError, match="unsupported record"):
        manager.embed({"bad": True})

    assert manager.is_started is True
    assert transport.stop_count == 0
    assert manager.embed({"bad": False})["output"]["input"] == {"bad": False}
    assert backend.prepare_calls == 1
    assert transport.start_count == 1
    assert transport.load_count == 1
    assert transport.embed_count == 2


def test_transport_failure_cleans_up_and_next_request_restarts(make_manager):
    transport = FakeTransport()
    transport.transport_failures.add(1)
    manager, backend, _ = make_manager(transport=transport)
    first_instance_name = manager.instance_name

    with pytest.raises(RuntimeError, match="relay exited"):
        manager.embed("first")

    assert manager.is_started is False
    assert transport.start_count == 1
    assert transport.load_count == 1
    assert transport.stop_count == 1
    assert manager.instance_name != first_instance_name
    restarted_instance_name = manager.instance_name

    result = manager.embed("second", kwargs={"retry": True})

    assert result["output"]["input"] == "second"
    assert result["output"]["kwargs"] == {"retry": True}
    assert manager.is_started is True
    assert backend.prepare_calls == 2
    assert transport.start_count == 2
    assert transport.load_count == 2
    assert transport.embed_count == 2
    assert transport.stop_count == 1
    assert manager.instance_name == restarted_instance_name


def test_failed_instance_start_can_be_retried(make_manager):
    transport = FakeTransport()
    manager, _, _ = make_manager(transport=transport)
    original_run = transport.run
    attempts = 0

    def fail_first_start(command, payload, environment, timeout):
        nonlocal attempts
        if command[0] == "start":
            attempts += 1
            if attempts == 1:
                raise RuntimeError("instance start failed")
        return original_run(command, payload, environment, timeout)

    manager._run_subprocess = fail_first_start

    with pytest.raises(RuntimeError, match="instance start failed"):
        manager.embed("first")
    assert manager.is_started is False
    assert transport.stop_count == 1

    assert manager.embed("second")["output"]["input"] == "second"
    assert attempts == 2
    assert transport.start_count == 1


def test_stale_image_error_explains_how_to_rebuild(make_manager):
    transport = MissingWorkerTransport()
    manager, backend, _ = make_manager(transport=transport)

    with pytest.raises(
        RuntimeError, match="built before persistent model workers"
    ) as exc:
        manager.embed("first")

    message = str(exc.value)
    assert backend.image_path in message
    assert (
        "vh-checker create-apptainer-image generic-model "
        "--output vhmodels-generic-model-persistent.sif"
    ) in message
    assert "image_path" in message
    assert "No module named vhmodels.vh_checker.worker" in str(exc.value.__cause__)
    assert manager.is_started is False
    assert transport.start_count == 1
    assert transport.stop_count == 1


def test_close_is_idempotent_and_closed_manager_cannot_restart(make_manager):
    manager, _, transport = make_manager()
    manager.embed("request")

    manager.close()
    manager.close()

    assert manager.is_started is False
    assert transport.stop_count == 1
    with pytest.raises(RuntimeError, match="manager is closed"):
        manager.embed("another request")
    assert transport.start_count == 1


def test_failed_stop_can_be_retried_without_forgetting_instance(make_manager):
    backend = FakeBackend()
    transport = FakeTransport()
    manager, _, _ = make_manager(backend=backend, transport=transport)
    manager.embed("request")
    original_run = transport.run
    stop_attempts = 0

    def flaky_stop(command, payload, environment, timeout):
        nonlocal stop_attempts
        if command[0] == "stop":
            stop_attempts += 1
            if stop_attempts == 1:
                raise RuntimeError("stop failed")
        return original_run(command, payload, environment, timeout)

    manager._run_subprocess = flaky_stop

    with pytest.raises(RuntimeError, match="stop failed"):
        manager.close()
    assert manager.is_started is True

    manager.close()
    assert stop_attempts == 2
    assert manager.is_started is False


def test_failed_cleanup_after_transport_error_blocks_duplicate_start(make_manager):
    transport = FakeTransport()
    transport.transport_failures.add(1)
    manager, _, _ = make_manager(transport=transport)
    original_run = transport.run

    def failing_stop(command, payload, environment, timeout):
        if command[0] == "stop":
            raise RuntimeError("stop unavailable")
        return original_run(command, payload, environment, timeout)

    manager._run_subprocess = failing_stop

    with pytest.raises(RuntimeError, match="relay exited"):
        manager.embed("first")
    assert transport.start_count == 1

    with pytest.raises(RuntimeError, match="could not be stopped"):
        manager.embed("second")
    assert transport.start_count == 1

    manager._run_subprocess = original_run
    manager.close()


def test_externally_removed_instance_can_restart(make_manager):
    transport = FakeTransport()
    manager, _, _ = make_manager(transport=transport)
    manager.embed("first")
    vanished_instance = manager.instance_name
    transport.transport_failures.add(2)
    transport.missing_instances.add(vanished_instance)
    original_run = transport.run

    def absent_stop(command, payload, environment, timeout):
        if command[0] == "stop" and command[1] == vanished_instance:
            raise RuntimeError("instance does not exist")
        return original_run(command, payload, environment, timeout)

    manager._run_subprocess = absent_stop

    with pytest.raises(RuntimeError, match="relay exited"):
        manager.embed("request after external stop")

    assert manager.is_started is False
    assert manager.instance_name != vanished_instance
    assert manager.embed("after restart")["output"]["input"] == "after restart"
    assert transport.start_count == 2
    assert transport.load_count == 2


@pytest.mark.parametrize("timeout", [None, 0, -1, float("nan"), float("inf")])
def test_manager_rejects_invalid_timeout(make_manager, timeout):
    with pytest.raises(ValueError, match="positive, finite"):
        make_manager(timeout=timeout)


def test_forked_process_cannot_manage_parent_instance(make_manager, monkeypatch):
    manager, _, transport = make_manager()
    manager.embed("request")
    monkeypatch.setattr(process_manager.os, "getpid", lambda: manager._creator_pid + 1)

    with pytest.raises(RuntimeError, match="after the process forks"):
        manager.embed("child request")
    manager._close_at_exit()
    assert transport.stop_count == 0


def test_modelproxy_context_closes_manager_after_exception(monkeypatch):
    recorded = {}

    class RecordingManager:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs
            recorded["close_calls"] = 0

        def close(self):
            recorded["close_calls"] += 1

    monkeypatch.setattr(factory, "get_backend", lambda runtime, env_name: FakeBackend())
    monkeypatch.setattr(factory, "ApptainerProcessManager", RecordingManager)

    class ContextFailure(Exception):
        pass

    proxy = factory.ModelProxy(
        project="generic-model",
        env_name="generic-model.sif",
        model="variant-a",
        runtime="apptainer",
        load_kwargs={"device": "cpu"},
    )
    with pytest.raises(ContextFailure):
        with proxy as entered:
            assert entered is proxy
            raise ContextFailure

    assert recorded["close_calls"] == 1


def test_instance_and_socket_names_are_safe_bounded_and_unique(
    make_manager, monkeypatch
):
    ids = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr(process_manager.os, "getpid", lambda: 4321)
    monkeypatch.setattr(
        process_manager.uuid,
        "uuid4",
        lambda: type("FakeUuid", (), {"hex": next(ids)})(),
    )
    project = "../../ Model name with spaces, punctuation! and 🧪" * 4

    first, _, _ = make_manager(project=project)
    second, _, _ = make_manager(project=project)

    assert first.instance_name != second.instance_name
    for manager in (first, second):
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", manager.instance_name)
        assert manager.instance_name.startswith("vhmodels-Model-name-with-spaces")
        assert manager.socket_path == f"/tmp/{manager.instance_name}.sock"
        # Linux sockaddr_un.sun_path is limited to 108 bytes including NUL.
        assert len(os.fsencode(manager.socket_path)) < 108
