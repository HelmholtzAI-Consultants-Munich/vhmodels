# Unit tests for the parent side of the parent<->child handover protocol:
# sentinel extraction and the process-group kill on timeout.
import json
import signal
import subprocess

import pytest

from vhmodels.vh_checker import embed as embed_runner
from vhmodels.vh_checker import factory
from vhmodels.vh_checker.factory import _extract_result, _run_subprocess, load_model
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    RESULT_MARKER,
)


def _frame(obj):
    return f"{RESULT_MARKER}{json.dumps(obj)}{RESULT_MARKER}\n"


# --- _extract_result -------------------------------------------------------


def test_extract_clean_frame():
    stdout = _frame({"output": [1, 2, 3]})
    assert _extract_result(stdout, "") == [1, 2, 3]


def test_extract_ignores_surrounding_noise():
    # tqdm / HuggingFace download bars / warnings around the real frame.
    stdout = (
        "Downloading model: 100%|####| 1.2G/1.2G\n"
        "Some warning with a stray { brace }\n"
        + _frame({"output": {"k": "v"}})
        + "trailing log line\n"
    )
    assert _extract_result(stdout, "") == {"k": "v"}


def test_extract_no_markers_fails():
    with pytest.raises(ValueError, match="no opening marker"):
        _extract_result("just some noise, no markers here", "")


def test_extract_missing_closing_marker_fails():
    # Opening marker present but truncated before the close (child killed).
    truncated = RESULT_MARKER + '{"output": [1, 2'
    with pytest.raises(ValueError, match="truncated"):
        _extract_result(truncated, "stderr detail")


def test_extract_non_json_between_markers_fails():
    bad = f"{RESULT_MARKER}not json{RESULT_MARKER}"
    with pytest.raises(ValueError, match="not valid JSON"):
        _extract_result(bad, "")


def test_extract_missing_output_key_fails():
    with pytest.raises(ValueError, match="missing 'output' key"):
        _extract_result(_frame({"something_else": 1}), "")


def test_extract_error_includes_stderr():
    with pytest.raises(ValueError, match="boom on stderr"):
        _extract_result("no markers", "boom on stderr")


def test_extract_does_not_double_wrap():
    # Model already returns an "output" envelope; result must be the inner value.
    result = _extract_result(_frame({"output": {"prediction": 42}}), "")
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
    monkeypatch.setattr(factory.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        factory.os, "killpg", lambda pgid, sig: recorded["signals"].append((pgid, sig))
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

    monkeypatch.setattr(factory.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(factory.os, "killpg", lambda pgid, sig: sent.append(sig))

    factory._terminate_group(FakeProc())

    # Graceful first, then forceful
    assert sent == [signal.SIGTERM, signal.SIGKILL]


# --- NDJSON request path ---------------------------------------------------


def test_modelproxy_embed_sends_ndjson_messages(monkeypatch):
    captured = {}

    def fake_run_subprocess(cmd, payload, subprocess_env, timeout):
        captured["cmd"] = cmd
        captured["payload"] = payload
        captured["subprocess_env"] = subprocess_env
        captured["timeout"] = timeout
        return (_frame({"output": {"prediction": 42}}), "")

    monkeypatch.setattr(factory, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        "vhmodels.vh_checker.backends.CondaBackend.is_available", lambda self: True
    )

    model = load_model("dinobloom", model="s", foo="bar", answer=42)
    result = model.embed({"text": "hello"})

    messages = [json.loads(line) for line in captured["payload"].splitlines()]
    assert messages == [
        {
            MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE,
            "load_kwargs": {"foo": "bar", "answer": 42},
        },
        {MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": {"text": "hello"}},
    ]
    assert result == {"prediction": 42}


def test_load_model_stores_load_kwargs():
    model = load_model("dinobloom", foo="bar")
    assert model.load_kwargs == {"foo": "bar"}


def test_child_dispatch_forwards_load_kwargs_and_embed_input():
    calls = []

    class FakeInstance:
        def load_model(self, model, **kwargs):
            calls.append(("load_model", model, kwargs))

        def embed(self, input):
            calls.append(("embed", input))
            return {"output": {"prediction": 7}}

    result = embed_runner._dispatch_embed(
        FakeInstance(),
        "s",
        [
            {MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE, "load_kwargs": {"foo": "bar"}},
            {MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": {"text": "hello"}},
        ],
    )

    assert calls == [
        ("load_model", "s", {"foo": "bar"}),
        ("embed", {"text": "hello"}),
    ]
    assert result == {"output": {"prediction": 7}}


def test_child_dispatch_treats_missing_load_kwargs_as_empty():
    calls = []

    class FakeInstance:
        def load_model(self, model, **kwargs):
            calls.append(("load_model", model, kwargs))

        def embed(self, input):
            calls.append(("embed", input))
            return {"output": input}

    result = embed_runner._dispatch_embed(
        FakeInstance(),
        None,
        [
            {MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE},
            {MESSAGE_TYPE_KEY: EMBED_MESSAGE_TYPE, "input": "hello"},
        ],
    )

    assert calls == [("load_model", None, {}), ("embed", "hello")]
    assert result == {"output": "hello"}


def test_child_dispatch_unknown_message_type_fails():
    class FakeInstance:
        def load_model(self, model, **kwargs):
            return None

    with pytest.raises(ValueError, match="Unknown message type 'predict'"):
        embed_runner._dispatch_embed(
            FakeInstance(),
            "s",
            [
                {MESSAGE_TYPE_KEY: LOAD_MESSAGE_TYPE, "load_kwargs": {}},
                {MESSAGE_TYPE_KEY: "predict", "input": "hello"},
            ],
        )
