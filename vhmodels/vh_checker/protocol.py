# Shared constants for the parent<->child handover protocol.
# Both the parent (ModelProxy in factory.py) and the child
# (vhmodels.vh_checker.embed) import these values so the contract cannot drift.
#
# Parent -> child request schema (newline-delimited JSON, one message per line):
# {"type": "load", "load_kwargs": {...}}
# {"type": "embed", "input": <any JSON value>}
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
