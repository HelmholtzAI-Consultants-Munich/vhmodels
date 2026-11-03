"""Shared constants and socket helpers for model worker communication."""

import json
import socket
import time


# Frames the relay's JSON reply on stdout, so other output is ignored.
RESULT_MARKER = "===VHMODELS_RESULT==="

# Request types; the worker replies {"ok": bool, "result" | "error": ...}.
MESSAGE_TYPE_KEY = "type"
LOAD_MESSAGE_TYPE = "load"
EMBED_MESSAGE_TYPE = "embed"
PREDICT_MESSAGE_TYPE = "predict"


def encode_message(message):
    return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")


def read_message(connection):
    with connection.makefile("r", encoding="utf-8", errors="replace") as stream:
        line = stream.readline()
    if not line:
        raise RuntimeError("The model worker closed the connection without a response.")
    return json.loads(line)


def send_request(
    socket_path,
    message,
    connect_timeout,
    response_timeout=None,
    is_running=None,
):
    """Send one JSON request to a worker and return its decoded response."""
    connection = None
    deadline = time.monotonic() + connect_timeout
    try:
        while True:
            if is_running is not None and not is_running():
                raise RuntimeError("The model worker exited before becoming ready.")
            candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                candidate.connect(socket_path)
                connection = candidate
                break
            except OSError as error:
                candidate.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Model worker socket '{socket_path}' was not ready within "
                        f"{connect_timeout:g}s."
                    ) from error
                time.sleep(0.05)

        connection.settimeout(response_timeout)
        connection.sendall(encode_message(message))
        connection.shutdown(socket.SHUT_WR)
        return read_message(connection)
    finally:
        if connection is not None:
            connection.close()
