# Unit tests for the parent side of the parent<->child handover protocol:
# sentinel extraction and the process-group kill on timeout.
import json
import signal
import subprocess

import pytest

from vhmodels.utils import lima_utils
from vhmodels.utils import subprocess_utils
from vhmodels.vh_checker import backends, factory
from vhmodels.vh_checker.factory import (
    _extract_frame,
    _run_subprocess,
    _unwrap_model_result,
    load_model,
)
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    PREDICT_MESSAGE_TYPE,
    RESULT_MARKER,
)


def _frame(obj):
    return f"{RESULT_MARKER}{json.dumps(obj)}{RESULT_MARKER}\n"


# --- result parsing --------------------------------------------------------


def test_extract_clean_frame():
    stdout = _frame({"output": [1, 2, 3]})
    assert _extract_frame(stdout, "") == {"output": [1, 2, 3]}


def test_extract_ignores_surrounding_noise():
    # tqdm / HuggingFace download bars / warnings around the real frame.
    stdout = (
        "Downloading model: 100%|####| 1.2G/1.2G\n"
        "Some warning with a stray { brace }\n"
        + _frame({"output": {"k": "v"}})
        + "trailing log line\n"
    )
    assert _extract_frame(stdout, "") == {"output": {"k": "v"}}


def test_extract_allows_marker_text_inside_model_data():
    value = f"before {RESULT_MARKER} after"
    assert _extract_frame(_frame({"output": value}), "") == {"output": value}


def test_extract_no_markers_fails():
    with pytest.raises(ValueError, match="no opening marker"):
        _extract_frame("just some noise, no markers here", "")


def test_extract_missing_closing_marker_fails():
    # Opening marker present but truncated before the close (child killed).
    truncated = RESULT_MARKER + '{"output": [1, 2'
    with pytest.raises(ValueError, match="truncated"):
        _extract_frame(truncated, "stderr detail")


def test_extract_non_json_between_markers_fails():
    bad = f"{RESULT_MARKER}not json{RESULT_MARKER}"
    with pytest.raises(ValueError, match="not valid JSON"):
        _extract_frame(bad, "")


def test_extract_missing_output_key_fails():
    with pytest.raises(ValueError, match="missing 'output' key"):
        _unwrap_model_result({"something_else": 1})


def test_extract_error_includes_stderr():
    with pytest.raises(ValueError, match="boom on stderr"):
        _extract_frame("no markers", "boom on stderr")


def test_extract_does_not_double_wrap():
    # Model already returns an "output" envelope; result must be the inner value.
    result = _unwrap_model_result({"output": {"prediction": 42}})
    assert result == {"prediction": 42}


# --- _run_subprocess -------------------------------------------------------


def test_run_subprocess_nonzero_exit_raises():
    # 'false' exits 1 with no output.
    with pytest.raises(RuntimeError, match="exit 1"):
        _run_subprocess(
            ["sh", "-c", "exit 1"], payload="", subprocess_env=None, timeout=10
        )


def test_run_subprocess_passes_stdin_and_captures_stdout():
    stdout, _ = _run_subprocess(
        ["cat"], payload="hello-stdin", subprocess_env=None, timeout=10
    )
    assert stdout == "hello-stdin"


def test_timeout_kills_process_group(monkeypatch):
    recorded = {"popen_kwargs": None, "signals": []}

    class FakeProc:
        pid = 4242
        returncode = -9

        def communicate(self, input=None, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return ("", "")  # drain call after the kill

        def wait(self, timeout=None):
            return 0  # dies immediately after the first signal

    def fake_popen(*args, **kwargs):
        recorded["popen_kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess_utils.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda pgid, sig: recorded["signals"].append((pgid, sig)),
    )

    with pytest.raises(RuntimeError, match="exceeded 1s"):
        _run_subprocess(["sleep", "100"], payload="", subprocess_env=None, timeout=1)

    # The child MUST be its own session/group leader. Without start_new_session,
    # getpgid(child) returns the parent's group and killpg would signal the
    # orchestrator itself -- so this assertion guards a dangerous regression.
    assert recorded["popen_kwargs"].get("start_new_session") is True

    # The signal targets the group (keyed by the leader pid), not just the child,
    # and starts with a graceful SIGTERM.
    assert recorded["signals"], "no signal was sent"
    assert all(pgid == FakeProc.pid for pgid, _ in recorded["signals"])
    assert recorded["signals"][0][1] == signal.SIGTERM


def test_terminate_group_escalates_sigterm_then_sigkill(monkeypatch):
    sent = []
    calls = {"wait": 0}

    class FakeProc:
        pid = 4242

        def wait(self, timeout=None):
            calls["wait"] += 1
            if calls["wait"] == 1:
                # SIGTERM ignored: the leader does not die within the grace window.
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return 0  # dies after SIGKILL

    monkeypatch.setattr(subprocess_utils.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        subprocess_utils.os, "killpg", lambda pgid, sig: sent.append(sig)
    )

    subprocess_utils.terminate_process_group(FakeProc())

    # Graceful first, then forceful
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_run_subprocess_interrupt_kills_process_group(monkeypatch):
    sent = []

    class FakeProc:
        pid = 4242
        returncode = -15
        calls = 0

        def communicate(self, input=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return ("", "")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(subprocess_utils.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        subprocess_utils.os, "killpg", lambda pgid, sig: sent.append((pgid, sig))
    )

    with pytest.raises(KeyboardInterrupt):
        _run_subprocess(["worker"], payload="", subprocess_env=None, timeout=10)

    assert sent == [(4242, signal.SIGTERM)]


# --- Persistent proxy path -------------------------------------------------


def test_modelproxy_conda_uses_one_manager_for_many_requests(monkeypatch, tmp_path):
    recorded = {"embed": [], "close": 0}

    class RecordingManager:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs

        def embed(self, **kwargs):
            recorded["embed"].append(kwargs)
            return {"output": len(recorded["embed"])}

        @property
        def is_started(self):
            return bool(recorded["embed"])

        def close(self):
            recorded["close"] += 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(factory, "CondaProcessManager", RecordingManager)
    monkeypatch.setattr(
        backends.CondaBackend, "is_runtime_available", lambda self: True
    )
    monkeypatch.setattr(backends.CondaBackend, "is_available", lambda self: True)

    model = load_model("dinobloom", model="s", device="cpu")
    assert model.embed("first", batch_size=2) == 1
    assert model.embed("second", batch_size=4) == 2

    assert recorded["init"]["project"] == "dinobloom"
    assert recorded["init"]["model"] == "s"
    assert recorded["init"]["load_kwargs"] == {"device": "cpu"}
    assert recorded["embed"] == [
        {"input": "first", "kwargs": {"batch_size": 2}, "cwd": tmp_path},
        {"input": "second", "kwargs": {"batch_size": 4}, "cwd": tmp_path},
    ]
    model.close()
    assert recorded["close"] == 1


def test_modelproxy_conda_predict_sends_embedding_to_manager(monkeypatch, tmp_path):
    recorded = {"embed": [], "predict": []}

    class RecordingManager:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs

        def embed(self, **kwargs):
            recorded["embed"].append(kwargs)
            return {"output": {"embedded": True}}

        def predict(self, **kwargs):
            recorded["predict"].append(kwargs)
            return {"output": {"probability": 0.9}}

        @property
        def is_started(self):
            return bool(recorded["embed"]) or bool(recorded["predict"])

        def close(self):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(factory, "CondaProcessManager", RecordingManager)
    monkeypatch.setattr(
        backends.CondaBackend, "is_runtime_available", lambda self: True
    )
    monkeypatch.setattr(backends.CondaBackend, "is_available", lambda self: True)

    model = load_model("mole")
    embedding = [[0.1, 0.2], [0.3, 0.4]]
    result = model.predict("molecules.tsv", embedding, strain="all")

    assert result == {"probability": 0.9}
    assert recorded["predict"] == [
        {
            "input": "molecules.tsv",
            "embedding": embedding,
            "kwargs": {"strain": "all"},
            "cwd": tmp_path,
        }
    ]


def test_modelproxy_apptainer_starts_loads_and_reuses_instance(monkeypatch, tmp_path):
    calls = []

    def fake_run_subprocess(cmd, payload, subprocess_env, timeout):
        calls.append((cmd, payload, subprocess_env, timeout))
        if "request" not in cmd:
            return ("", "")
        message = json.loads(payload)
        result = None
        if message[MESSAGE_TYPE_KEY] == EMBED_MESSAGE_TYPE:
            result = {"output": [message["input"]]}
        return (_frame({"ok": True, "result": result}), "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lima_utils, "uses_lima", lambda: False)
    monkeypatch.setattr(factory, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        "vhmodels.vh_checker.backends.ApptainerBackend.is_available",
        lambda self: True,
    )
    monkeypatch.setattr(
        "vhmodels.vh_checker.backends.ApptainerBackend.is_runtime_available",
        lambda self: True,
    )

    model = load_model("dinobloom", model="s", runtime="apptainer", device="cpu")
    assert model.embed("first.bmp") == ["first.bmp"]
    assert model.embed("second.bmp", batch_size=4) == ["second.bmp"]

    assert calls[0][0][:4] == [
        "apptainer",
        "instance",
        "start",
        str(tmp_path / "vhmodels-dinobloom.sif"),
    ]
    request_messages = [
        json.loads(payload) for cmd, payload, _, _ in calls if "request" in cmd
    ]
    assert request_messages == [
        {
            MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
            "project": "dinobloom",
            "model": "s",
            "load_kwargs": {"device": "cpu"},
        },
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": "first.bmp",
            "kwargs": {},
            "cwd": str(tmp_path),
        },
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": "second.bmp",
            "kwargs": {"batch_size": 4},
            "cwd": str(tmp_path),
        },
    ]
    assert sum(cmd[:3] == ["apptainer", "instance", "start"] for cmd, *_ in calls) == 1
    assert all("PYTHONPATH" not in env for _, _, env, _ in calls)

    model.close()
    assert calls[-1][0][:3] == ["apptainer", "instance", "stop"]


def test_modelproxy_apptainer_predict_uses_embed_output(monkeypatch, tmp_path):
    calls = []

    def fake_run_subprocess(cmd, payload, subprocess_env, timeout):
        calls.append((cmd, payload, subprocess_env, timeout))
        if "request" not in cmd:
            return ("", "")
        message = json.loads(payload)
        result = None
        if message[MESSAGE_TYPE_KEY] == EMBED_MESSAGE_TYPE:
            result = {"output": [[0.1, 0.2]]}
        elif message[MESSAGE_TYPE_KEY] == PREDICT_MESSAGE_TYPE:
            result = {"output": {"embedding": message["embedding"]}}
        return (_frame({"ok": True, "result": result}), "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lima_utils, "uses_lima", lambda: False)
    monkeypatch.setattr(factory, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        "vhmodels.vh_checker.backends.ApptainerBackend.is_available",
        lambda self: True,
    )
    monkeypatch.setattr(
        "vhmodels.vh_checker.backends.ApptainerBackend.is_runtime_available",
        lambda self: True,
    )

    model = load_model("mole", runtime="apptainer", device="cpu")
    embedding = model.embed("sequences.smiles")
    result = model.predict("molecules.tsv", embedding)

    assert result == {"embedding": [[0.1, 0.2]]}
    request_messages = [
        json.loads(payload) for cmd, payload, _, _ in calls if "request" in cmd
    ]
    assert request_messages[-1] == {
        MESSAGE_TYPE_KEY: PREDICT_MESSAGE_TYPE,
        "input": "molecules.tsv",
        "embedding": [[0.1, 0.2]],
        "kwargs": {},
        "cwd": str(tmp_path),
    }
    model.close()


def test_apptainer_image_path_can_be_overridden(tmp_path):
    image_path = tmp_path / "custom.sif"
    model = load_model("mole", runtime="apptainer", image_path=image_path)
    assert model.env_name == str(image_path.resolve())
    assert model.backend.image_path == str(image_path.resolve())


def test_missing_apptainer_image_has_runtime_specific_guidance(tmp_path):
    image_path = tmp_path / "missing.sif"
    model = load_model("mole", runtime="apptainer", image_path=image_path)
    with pytest.raises(RuntimeError, match="create-apptainer-image mole"):
        model.embed("sequences.smiles")


def test_missing_apptainer_executable_has_runtime_specific_guidance(
    monkeypatch, tmp_path
):
    image_path = tmp_path / "mole.sif"
    image_path.touch()
    monkeypatch.setattr(lima_utils, "uses_lima", lambda: False)
    model = load_model("mole", runtime="apptainer", image_path=image_path)
    monkeypatch.setattr(model.backend, "is_runtime_available", lambda: False)
    with pytest.raises(RuntimeError, match="executable is not available"):
        model.embed("sequences.smiles")


def test_modelproxy_apptainer_prepares_and_uses_lima(monkeypatch, tmp_path):
    captured = {"calls": []}
    image_path = tmp_path / "dinobloom.sif"
    image_path.touch()

    def fake_run_subprocess(cmd, payload, subprocess_env, timeout):
        captured["calls"].append((cmd, payload))
        if "request" in cmd:
            message = json.loads(payload)
            result = (
                {"output": [42]}
                if message[MESSAGE_TYPE_KEY] == EMBED_MESSAGE_TYPE
                else None
            )
            return (_frame({"ok": True, "result": result}), "")
        return ("", "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lima_utils, "uses_lima", lambda: True)
    monkeypatch.setattr(lima_utils, "is_lima_shared_path", lambda path: True)
    monkeypatch.setattr(factory, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        backends.ApptainerBackend,
        "is_runtime_available",
        lambda self: True,
    )
    monkeypatch.setattr(
        backends.ApptainerBackend,
        "prepare",
        lambda self: captured.setdefault("prepared", True),
    )

    model = load_model(
        "dinobloom", runtime="apptainer", image_path=image_path, device="cpu"
    )
    result = model.embed("image.bmp")

    assert captured["prepared"] is True
    assert all(
        command[:7]
        == [
            "limactl",
            "shell",
            "--tty=false",
            "--preserve-env",
            "--workdir",
            str(backends.Path.home().resolve()),
            lima_utils.LIMA_INSTANCE,
        ]
        for command, _ in captured["calls"]
    )
    assert captured["calls"][0][0][7:10] == [
        "apptainer",
        "instance",
        "start",
    ]
    assert [
        json.loads(payload)
        for command, payload in captured["calls"]
        if "request" in command
    ] == [
        {
            MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
            "project": "dinobloom",
            "model": None,
            "load_kwargs": {"device": "cpu"},
        },
        {
            MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE,
            "input": "image.bmp",
            "kwargs": {},
            "cwd": str(tmp_path.resolve()),
        },
    ]
    assert result == [42]
    model.close()


def test_image_path_is_rejected_for_conda(tmp_path):
    with pytest.raises(ValueError, match="runtime='apptainer'"):
        load_model("mole", image_path=tmp_path / "mole.sif")


def test_load_model_stores_load_kwargs():
    model = load_model("dinobloom", foo="bar")
    assert model.load_kwargs == {"foo": "bar"}
