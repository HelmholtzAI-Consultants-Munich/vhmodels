"""Persistent model worker used by isolated runtime processes.

The server owns the model object and communicates over a Unix-domain socket.
Apptainer launches the request CLI as a lightweight relay, while Conda connects
to the same socket protocol directly from the host process.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import sys
import traceback

from vhmodels.vh_checker.base import BaseModel
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    RESULT_MARKER,
    encode_message,
    read_message,
    send_request,
)


@contextmanager
def _working_directory(path):
    """Temporarily enter the caller's directory and restore it reliably."""
    if path is None:
        yield
        return

    previous_directory = os.open(".", os.O_RDONLY)
    try:
        os.chdir(path)
        yield
    finally:
        os.fchdir(previous_directory)
        os.close(previous_directory)


class ModelWorker:
    """Model-agnostic request dispatcher that loads exactly one model."""

    def __init__(self):
        self._model = None

    @property
    def is_loaded(self):
        return self._model is not None

    def handle(self, message):
        if not isinstance(message, dict):
            raise ValueError("Worker requests must be JSON objects.")

        message_type = message.get(MESSAGE_TYPE_KEY)
        if message_type == LOAD_MESSAGE_TYPE:
            return self._load(message)
        if message_type == EMBED_MESSAGE_TYPE:
            return self._embed(message)
        raise ValueError(f"Unknown message type '{message_type}'.")

    def _load(self, message):
        if self.is_loaded:
            raise RuntimeError("This worker already has a loaded model.")

        project = message.get("project")
        if not isinstance(project, str) or not project:
            raise ValueError("A non-empty project is required to load a model.")

        load_kwargs = message.get("load_kwargs") or {}
        if not isinstance(load_kwargs, dict):
            raise ValueError("load_kwargs must be a JSON object.")

        model_class = BaseModel.get_class(project)
        model = model_class()
        model.load_model(message.get("model"), **load_kwargs)
        # Publish only a completely loaded model. A failed load leaves the
        # worker unloaded so the process manager can tear the instance down.
        self._model = model
        return None

    def _embed(self, message):
        if not self.is_loaded:
            raise RuntimeError("The model must be loaded before embedding.")

        kwargs = message.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a JSON object.")

        with _working_directory(message.get("cwd")):
            return self._model.embed(message.get("input"), **kwargs)


def serve(socket_path):
    """Serve requests sequentially until the Apptainer instance is stopped."""
    path = Path(socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound = False
    previous_handlers = {}

    def stop_worker(signum, frame):
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop_worker)
        server.bind(os.fspath(path))
        bound = True
        os.chmod(path, 0o600)
        server.listen(1)
        worker = ModelWorker()

        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    request = read_message(connection)
                    result = worker.handle(request)
                    response = {"ok": True, "result": result}
                    encoded_response = encode_message(response)
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    encoded_response = encode_message(response)
                try:
                    connection.sendall(encoded_response)
                except OSError:
                    # The host may have timed out and killed its relay. The
                    # process manager will stop this instance in that case.
                    pass
    finally:
        server.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if bound:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def request(socket_path, connect_timeout):
    """Relay one stdin JSON request to a running worker and frame its reply."""
    message = json.loads(sys.stdin.read())
    response = send_request(socket_path, message, connect_timeout)

    sys.stdout.write(
        RESULT_MARKER + json.dumps(response, ensure_ascii=False) + RESULT_MARKER + "\n"
    )
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    # ``instance start`` treats option-looking trailing arguments as its own
    # flags, so the startscript receives the socket as a plain positional value.
    serve_parser.add_argument("socket")

    request_parser = subparsers.add_parser("request")
    request_parser.add_argument("--socket", required=True)
    request_parser.add_argument("--connect-timeout", type=float, default=30.0)

    args = parser.parse_args()
    try:
        if args.command == "serve":
            serve(args.socket)
        else:
            request(args.socket, args.connect_timeout)
    except KeyboardInterrupt:
        return
    except Exception as error:
        print(f"Model Worker Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
