# Shared constants for the parent<->child handover protocol.
# Both the parent (ModelProxy in factory.py) and the child
# (vhmodels.vh_checker.embed) import these values so the contract cannot drift.
#
# One-shot parent -> child request schema (newline-delimited JSON):
# {"type": "load", "load_kwargs": {...}}
# {"type": "embed", "input": <any JSON value>, "kwargs": {...}}
#
# Persistent-worker requests use the same message types. The initial load also
# carries project/model, while each subsequent embed may carry cwd. The worker
# replies with {"ok": true, "result": ...} or {"ok": false, "error": ...}.
#
# Child -> parent response schema:
# RESULT_MARKER + json.dumps(<model result dict>) + RESULT_MARKER + "\n"
#
# Everything else on stdout (progress bars, library logging, warnings) is
# ignored by the parser, so the result channel is immune to that noise.

RESULT_MARKER = "===VHMODELS_RESULT==="

MESSAGE_TYPE_KEY = "type"
LOAD_MESSAGE_TYPE = "load"
EMBED_MESSAGE_TYPE = "embed"
