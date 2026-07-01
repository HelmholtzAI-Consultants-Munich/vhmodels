# The file contains the necessary to run the function 'embed' for the given model.
# It runs *inside* the model's isolated environment as a subprocess. The parent
# (ModelProxy) sends newline-delimited JSON messages on stdin and reads the result
# framed between RESULT_MARKERs on stdout.
from vhmodels.vh_checker.base import BaseModel
from vhmodels.vh_checker.protocol import (
    EMBED_MESSAGE_TYPE,
    LOAD_MESSAGE_TYPE,
    MESSAGE_TYPE_KEY,
    RESULT_MARKER,
)

import argparse
import json
import sys


def _parse_request_messages(payload):
    messages = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        messages.append(json.loads(line))
    return messages


def _require_message_type(message, expected_type):
    message_type = message.get(MESSAGE_TYPE_KEY)
    if message_type != expected_type:
        raise ValueError(
            f"Expected message type '{expected_type}', got '{message_type}'."
        )
    return message


def _dispatch_embed(instance, model_name, messages):
    if len(messages) < 2:
        raise ValueError(
            "Expected exactly one load message followed by one embed message."
        )

    load_message = _require_message_type(messages[0], LOAD_MESSAGE_TYPE)
    load_kwargs = load_message.get("load_kwargs") or {}
    instance.load_model(model_name, **load_kwargs)

    operation_message = messages[1]
    operation_type = operation_message.get(MESSAGE_TYPE_KEY)
    if operation_type == EMBED_MESSAGE_TYPE:
        return instance.embed(operation_message.get("input"))
    raise ValueError(f"Unknown message type '{operation_type}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="ID of the project to run")
    parser.add_argument("--model", required=False, help="Exact model in the project")
    args = parser.parse_args()

    model_cls = BaseModel.get_class(args.project)

    try:
        # Read tagged request messages from stdin. Using stdin (rather than a
        # CLI argument) avoids the OS argument-length limit on large inputs.
        payload = sys.stdin.read()
        messages = _parse_request_messages(payload)

        # Instantiate and load weights (this happens inside the sub-env).
        instance = model_cls()
        result = _dispatch_embed(instance, args.model, messages)

        # Frame the result so the parent can extract it regardless of any other
        # output libraries may have written to stdout.
        sys.stdout.write(RESULT_MARKER + json.dumps(result) + RESULT_MARKER + "\n")
        sys.stdout.flush()

    except Exception as e:
        # Errors go to stderr so the parent can report them; a non-zero exit
        # tells the parent the run failed.
        print(f"Internal Model Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
